import os
import time
import random
import itertools

import pandas as pd
import yaml
import enum
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torchio as tio

from models import UnetEncoderSSLRLP
from utils import (
    get_subjects,
    get_train_valid_data_loaders,
    replace_target_norm_layer_for_group_norm
)
from logger import NeptuneLogger


class GetPatches:
    def __init__(self, patch_dim, gap):
        self.patch_dim = patch_dim
        self.gap = gap
        self.patch_loc_arr = [
            (1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)
        ]

    def __call__(self, sample):
        volume = sample.spect.data

        offset_x = volume.shape[1] - (self.patch_dim * 3 + self.gap * 2)
        offset_y = volume.shape[2] - (self.patch_dim * 3 + self.gap * 2)

        start_grid_x = np.random.randint(0, offset_x)
        start_grid_y = np.random.randint(0, offset_y)

        random_patch_label = np.random.randint(len(self.patch_loc_arr))
        tempx, tempy = self.patch_loc_arr[random_patch_label]

        rp_x_pt = start_grid_x + self.patch_dim * (
                tempx - 1) + self.gap * (tempx - 1)
        rp_y_pt = start_grid_y + self.patch_dim * (
                tempy - 1) + self.gap * (tempy - 1)
        random_patch = volume[
           :,
           rp_x_pt: rp_x_pt + self.patch_dim,
           rp_y_pt: rp_y_pt + self.patch_dim,
           :
        ]
        cp_x_pt = start_grid_x + self.patch_dim + self.gap
        cp_y_pt = start_grid_y + self.patch_dim + self.gap
        center_patch = volume[
           :,
           cp_x_pt: cp_x_pt + self.patch_dim,
           cp_y_pt: cp_y_pt + self.patch_dim,
           :
        ]
        random_patch_label = torch.tensor(random_patch_label, dtype=torch.long)

        return center_patch, random_patch, random_patch_label


class NormalizePatches:
    def __init__(self):
        self.normalize = tio.ZNormalization()

    def __call__(self, sample):
        center_patch, random_patch, random_patch_label = sample

        try:
            cp = self.normalize(center_patch)
            rp = self.normalize(random_patch)

            return cp, rp, random_patch_label
        except:
            cp_cond = torch.all(center_patch == 0)
            rp_cond = torch.all(random_patch == 0)
            cp = self.normalize(center_patch) if not cp_cond else center_patch
            rp = self.normalize(random_patch) if not rp_cond else random_patch

            return cp, rp, random_patch_label


def get_augmentation_pipelines(
    target_shape,
    max_displacement,
    num_control_points,
    patch_dim,
    gap,
    **kwargs
):
    """

    :param target_shape:
    :param max_displacement:
    :param num_control_points:
    :param patch_dim:
    :param gap:
    :param kwargs:
    :return:
    """
    training_transform = tio.Compose([
        tio.CropOrPad(target_shape=target_shape),
        tio.RandomFlip(axes=(0, 1, 2), p=0.5),
        tio.OneOf({
            tio.RandomAnisotropy(): 0.5,
            tio.RandomBlur(): 0.5,
            tio.RandomNoise(): 0.5,
        }, p=0.5),
        tio.OneOf({
            tio.RandomAffine(): 0.5,
            tio.RandomElasticDeformation(
                max_displacement=max_displacement): 0.5,
            tio.RandomElasticDeformation(
                max_displacement=max_displacement,
                num_control_points=num_control_points
            ): 0.5,
            tio.RandomMotion(num_transforms=4,
                             image_interpolation='linear'): 0.5
        }, p=0.8),
        GetPatches(patch_dim, gap),
        NormalizePatches()
    ])
    validation_transform = tio.Compose([
        tio.CropOrPad(target_shape=target_shape),
        GetPatches(patch_dim, gap),
        NormalizePatches()
    ])
    return training_transform, validation_transform


class Action(enum.Enum):
    TRAIN = 'Training'
    VALIDATE = 'Validation'


def prepare_batch(batch, device):
    """

    :param batch:
    :param device:
    :return:
    """
    inputs1, inputs2, targets = batch
    inputs1 = inputs1.to(device)
    inputs2 = inputs2.to(device)
    targets = targets.to(device)

    return inputs1, inputs2, targets


def run_epoch(device, action, loader, model, criterion, optimizer):
    """

    :param device:
    :param action:
    :param loader:
    :param model:
    :param criterion:
    :param optimizer:
    :return:
    """
    is_training = action == Action.TRAIN
    epoch_losses = []
    accuracies = []
    model.train(is_training)

    for batch_idx, batch in enumerate(tqdm(loader)):

        inputs1, inputs2, targets = prepare_batch(batch, device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            # CrossEntropyLoss combines LogSoftmax with NLLLOSS
            outputs = model(inputs1, inputs2)
            batch_losses = criterion(outputs, targets)
            batch_loss = batch_losses.mean()
            accuracy = (
                (outputs.argmax(axis=1) == targets).sum() / len(outputs)
            ).cpu().item()
            accuracies.append(accuracy)

            if is_training:
                batch_loss.backward()
                optimizer.step()

            epoch_losses.append(batch_loss.item())

    epoch_losses = np.array(epoch_losses).mean()
    accuracies = np.array(accuracies).mean()
    print(
        f'{action.value} mean loss: {epoch_losses.mean():0.3f}\tAccuracy: '
        f'{accuracies:.2f}'
    )

    return time.time(), epoch_losses, accuracies


def train(
    num_epochs,
    training_loader,
    validation_loader,
    model,
    criterion,
    optimizer,
    device,
    path_save,
    early_stop
):
    """

    :param num_epochs:
    :param training_loader:
    :param validation_loader:
    :param model:
    :param criterion:
    :param optimizer:
    :param device:
    :param path_save:
    :param early_stop:
    :return:
    """
    train_time, train_loss, train_acc = [], [], []
    valid_time, valid_loss, valid_acc = [], [], []
    best_val_acc = -np.inf
    early_stop_counter = 0

    for epoch_idx in tqdm(range(1, num_epochs + 1)):
        print('Starting epoch', epoch_idx)
        tr_time, tr_loss, tr_acc = run_epoch(
            device, Action.TRAIN, training_loader, model,
            criterion, optimizer
        )
        train_time.append(tr_time)
        train_loss.append(tr_loss)
        train_acc.append(tr_acc)

        vl_time, vl_loss, vl_acc = run_epoch(
            device, Action.VALIDATE, validation_loader, model,
            criterion, optimizer
        )
        valid_time.append(vl_time)
        valid_loss.append(vl_loss)
        valid_acc.append(vl_acc)

        if valid_acc[-1] > best_val_acc:
            print(f'Saving model at epoch: {epoch_idx}')
            best_val_acc = valid_acc[-1]
            early_stop_counter = 0
            checkpoint = {
                'train_losses': train_loss,
                'val_losses': valid_loss,
                'weights': model.state_dict()
            }
            torch.save(checkpoint, path_save)
        else:
            early_stop_counter += 1

            if early_stop_counter >= early_stop:
                print(f'Stopping after {epoch_idx} iterations...')
                break

    results = dict(zip(
        ['train_time', 'train_loss', 'train_acc',
         'valid_time', 'valid_loss', 'valid_acc'],
        [train_time, train_loss, train_acc, valid_time, valid_loss, valid_acc]
    ))
    return results


def run_experiment(conf, path_conf):
    """

    :param conf:
    :return:
    """
    seed = conf['seed']
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    nl = NeptuneLogger(
        parameters=dict(filter(lambda x: x[0] != 'neptune', conf.items())),
        **conf['neptune']
    )
    path_saved_models = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        'saved_models'
    )
    path_save = os.path.join(
        path_saved_models,
        f"{conf['experiment_name']}.pth"
    )
    path_artifacts = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        'artifacts',
        conf['experiment_name']
    )
    path_df_tr = os.path.join(path_artifacts, 'results.csv')

    if not os.path.isdir(path_artifacts):
        os.mkdir(path_artifacts)

    train_transform, valid_transform = get_augmentation_pipelines(
        **conf['augmentation_pipeline']
    )
    subjects = get_subjects(conf['file_spect_data'], conf['folder_volumes'])
    training_loader, validation_loader = get_train_valid_data_loaders(
        subjects,
        conf['training_split_ratio'],
        train_transform,
        valid_transform,
        conf['data_loader']['batch_size'],
        conf['data_loader']['batch_size'] * 2,
        conf['data_loader']['num_workers']
    )
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    model = UnetEncoderSSLRLP(**conf['model'])

    if conf['group_norm']['replace']:
        replace_target_norm_layer_for_group_norm(
            model,
            nn.InstanceNorm3d,
            nn.GroupNorm,
            num_groups=conf['group_norm']['groups']
        )

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=conf['optimizer']['learning_rate']
    )
    criterion = nn.CrossEntropyLoss()

    train_results = train(
        training_loader=training_loader,
        validation_loader=validation_loader,
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        path_save=path_save,
        **conf['train']
    )

    df_tr = pd.DataFrame(train_results)
    df_tr.to_csv(path_df_tr, index=False)

    nl.log_metrics(train_results)
    nl.log_artifacts([
        path_save, path_df_tr, path_conf
    ])


if __name__ == '__main__':

    path_experiment_conf = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        'experiments',
        'ssl_relative_patch_location.yaml'
    )
    with open(path_experiment_conf, 'r') as file:
        conf = yaml.load(file, Loader=yaml.Loader)

    run_experiment(conf, path_experiment_conf)
