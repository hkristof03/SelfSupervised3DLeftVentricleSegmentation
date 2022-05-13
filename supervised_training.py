import os
import random
import time

import yaml
import enum
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import torch
import torchio as tio
import torch.nn.functional as F
from torchvision.ops.focal_loss import sigmoid_focal_loss
from unet import UNet

from utils import (
    get_annotated_subjects,
    get_train_valid_data_loaders,
    Visualizer
)
from logger import NeptuneLogger


CHANNELS_DIMENSION = 1
SPATIAL_DIMENSIONS = (2, 3, 4)
FOREGROUND = 1
FIRST = 1


class Action(enum.Enum):
    TRAIN = 'Training'
    VALIDATE = 'Validation'


def prepare_batch(batch, device):
    """

    :param batch:
    :param device:
    :return:
    """
    inputs = batch['spect'][tio.DATA].to(device)
    targets = batch['left_ventricle'][tio.DATA].to(device)

    return inputs, targets


def get_dice_score(output, target, epsilon=1e-9):
    """

    :param output:
    :param target:
    :param epsilon:
    :return:
    """
    p0 = output
    g0 = target
    p1 = 1 - p0
    g1 = 1 - g0
    tp = (p0 * g0).sum(dim=SPATIAL_DIMENSIONS)
    fp = (p0 * g1).sum(dim=SPATIAL_DIMENSIONS)
    fn = (p1 * g0).sum(dim=SPATIAL_DIMENSIONS)
    num = 2 * tp
    denom = 2 * tp + fp + fn + epsilon
    dice_score = num / denom

    return dice_score


def get_dice_loss(output, target):

    return 1 - get_dice_score(output, target)


def get_log_cosh_dice_loss(output, target):

    return torch.log(torch.cosh(get_dice_loss(output, target)))


def compute_metrics(prediction, target):
    """

    :param prediction:
    :param target:
    :return:
    """
    pred = prediction.argmax(dim=1)
    targ = target.argmax(dim=1)
    p1 = 1 - pred
    g1 = 1 - targ

    tp = (targ * pred).sum(dim=(1, 2, 3))
    fp = (pred * g1).sum(dim=(1, 2, 3))
    fn = (p1 * targ).sum(dim=(1, 2, 3))

    precision = (tp / (tp + fp)).mean().cpu().numpy().item()
    recall = (tp / (tp + fn)).mean().cpu().numpy().item()
    iou = (tp / (tp + fp + fn)).mean().cpu().numpy().item()

    return precision, recall, iou


def get_augmentation_pipelines(
    target_shape,
    max_displacement,
    num_control_points,
    **kwargs
):
    """

    :param target_shape:
    :param max_displacement:
    :param num_control_points:
    :param kwargs:
    :return:
    """
    training_transform = tio.Compose([
        tio.CropOrPad(target_shape=target_shape, mask_name="left_ventricle"),
        tio.ZNormalization(),
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
        tio.OneHot()
    ])
    validation_transform = tio.Compose([
        tio.CropOrPad(target_shape=target_shape, mask_name="left_ventricle"),
        tio.ZNormalization(),
        tio.OneHot()
    ])
    return training_transform, validation_transform


def get_model_and_optimizer(config, device):
    """

    :param config:
    :param device:
    :return:
    """
    model = UNet(
        in_channels=1,
        out_classes=2,
        dimensions=3,
        upsampling_type='linear',
        padding=True,
        activation='PReLU',
        **config['model']['UNet']
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['optimizer']['learning_rate']
    )

    return model, optimizer


def get_criterion(conf):

    criterion = conf['criterion']['name']

    if criterion == 'dice':
        return get_dice_loss
    elif criterion == 'focal':
        gamma = conf['criterion']['gamma']
        return lambda *args: sigmoid_focal_loss(*args, gamma=gamma)
    elif criterion == 'logcosh':
        return get_log_cosh_dice_loss
    else:
        print('Returning default loss function - Dice Loss...')
        return get_dice_loss


def train(
    num_epochs,
    training_loader,
    validation_loader,
    model,
    criterion,
    optimizer,
    device,
    path_save,
    early_stop,
    scheduler=None
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
    :param scheduler:
    :return:
    """
    train_stats = []
    valid_stats = []
    best_val_iou = -np.inf
    early_stop_counter = 0

    for epoch_idx in tqdm(range(1, num_epochs + 1)):
        print('Starting epoch', epoch_idx)
        train_stats.append(run_epoch(
            device, Action.TRAIN, training_loader, model,
            criterion, optimizer, scheduler
        ))

        valid_stats.append(run_epoch(
            device, Action.VALIDATE, validation_loader, model,
            criterion, optimizer, scheduler
        ))
        vl_iou = valid_stats[-1][-1]

        if vl_iou > best_val_iou:
            print(f'Saving model at epoch: {epoch_idx}')
            best_val_iou = vl_iou
            early_stop_counter = 0
            checkpoint = {
                'weights': model.state_dict()
            }
            torch.save(checkpoint, path_save)
        else:
            early_stop_counter += 1

            if early_stop_counter >= early_stop:
                print(f'Stopping after {epoch_idx} iterations...')
                break

    train_cols = [
        'train_time', 'train_loss', 'train_precision', 'train_recall',
        'train_iou'
    ]
    valid_cols = [
        'valid_time', 'valid_loss', 'valid_precision', 'valid_recall',
        'valid_iou'
    ]
    df_train_res = pd.DataFrame(data=train_stats, columns=train_cols)
    df_valid_res = pd.DataFrame(data=valid_stats, columns=valid_cols)
    df_results = pd.concat([df_train_res, df_valid_res], axis=1)

    return df_results


def run_epoch(
    device,
    action,
    loader,
    model,
    criterion,
    optimizer,
    scheduler,
):
    """

    :param device:
    :param action:
    :param loader:
    :param model:
    :param criterion:
    :param optimizer:
    :param scheduler:
    :return:
    """
    is_training = action == Action.TRAIN
    epoch_stats = []
    model.train(is_training)

    for batch_idx, batch in enumerate(tqdm(loader)):

        inputs, targets = prepare_batch(batch, device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):

            logits = model(inputs)
            probabilities = F.softmax(logits, dim=CHANNELS_DIMENSION)
            batch_losses = criterion(probabilities, targets)
            batch_loss = batch_losses.mean()

            if is_training:
                batch_loss.backward()
                optimizer.step()

            prec, rec, iou, = compute_metrics(probabilities, targets)
            epoch_stats.append([batch_loss.item(), prec, rec, iou])

    if scheduler is not None:
        scheduler.step()

    epoch_stats = np.array(epoch_stats).mean(axis=0).tolist()

    print(
        f'{action.value} mean loss: {epoch_stats[0]:0.3f}\tPrecision: '
        f'{epoch_stats[1]:.2f}\tRecall: {epoch_stats[2]:.2f}\tIoU: '
        f'{epoch_stats[3]:.2f}'
    )

    return [time.time()] + epoch_stats


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
    subjects = get_annotated_subjects(
        conf['file_spect_data'],
        conf['folder_volumes']
    )
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

    model, optimizer = get_model_and_optimizer(conf, device)
    criterion = get_criterion(conf)
    scheduler = getattr(
        torch.optim.lr_scheduler,
        conf['scheduler']['name']
    )(optimizer, **conf['scheduler']['params']) if conf['scheduler'] else None

    if conf['model']['pretrained']:
        checkpoint = torch.load(
            os.path.join(
                path_saved_models,
                f"{conf['model']['experiment_name']}.pth"
            )
        )
        encoder_weights = {
            k.replace('encoder.', ''): v for k, v in checkpoint[
                'weights'].items() if k.startswith('encoder')
        }
        model.encoder.load_state_dict(encoder_weights)

        blocks_to_freeze = conf['model']['blocks_to_freeze']

        for name, param in model.named_parameters():
            if any([''.join(['encoding_', x]) in name for x in blocks_to_freeze]):
                param.requires_grad = False
            else:
                param.requires_grad = True

    df_results = train(
        conf['train']['num_epochs'],
        training_loader,
        validation_loader,
        model,
        criterion,
        optimizer,
        device,
        path_save,
        conf['train']['early_stop'],
        scheduler
    )
    df_results.to_csv(path_df_tr, index=False)

    nl.log_metrics(
        dict(zip(
            df_results.columns,
            df_results.values.T.tolist()
        ))
    )
    nl.log_artifacts([
        path_save, path_df_tr, path_conf
    ])

    vis = Visualizer()
    best_weights = torch.load(path_save)['weights']
    model.load_state_dict(best_weights)
    model.eval()
    pred_counter = 0

    for batch_idx, batch in enumerate(tqdm(validation_loader)):

        inputs, targets = prepare_batch(batch, device)

        with torch.no_grad():

            predictions = model(inputs).softmax(dim=1)
            probabilities = predictions[:, FOREGROUND:].cpu()

        for i in range(len(batch['spect'][tio.DATA])):

            label = batch['left_ventricle'][tio.DATA][i][1:, ...].permute(
                3, 0, 1, 2
            )
            pred = probabilities[i].permute(3, 0, 1, 2)
            vis.visualize(
                np.squeeze(label.permute(1, 0, 2, 3).numpy(), axis=0),
                np.squeeze(pred.permute(1, 0, 2, 3).numpy(), axis=0),
                os.path.join(
                    path_artifacts, f'mask_predictions_{pred_counter}.png'
                )
            )

            affine = batch['spect'][tio.AFFINE][i].numpy()
            subject = tio.Subject(
                spect=tio.ScalarImage(
                    tensor=batch['spect'][tio.DATA][i], affine=affine
                ),
                label=tio.LabelMap(
                    tensor=batch['left_ventricle'][tio.DATA][i], affine=affine
                ),
                predicted=tio.ScalarImage(
                    tensor=probabilities[i], affine=affine
                )
            )
            output_path = os.path.join(
                path_artifacts, f'predictions_planes_view_{pred_counter}.png'
            )
            subject.plot(
                figsize=(9, 8),
                cmap_dict={'predicted': 'RdBu_r'},
                show=False,
                output_path=output_path
            )
            pred_counter += 1


if __name__ == '__main__':

    path_experiment_conf = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        'experiments',
        'supervised_training.yaml'
    )
    with open(path_experiment_conf, 'r') as file:
        conf = yaml.load(file, Loader=yaml.Loader)

    run_experiment(conf, path_experiment_conf)
