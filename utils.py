import os
import warnings

import numpy as np
import nrrd
import pandas as pd
from skimage.util import montage
import torch
import torchio as tio
import matplotlib.pyplot as plt


def load_image(path):
    data, header = nrrd.read(path)
    data = data.astype(np.float32)
    affine = np.eye(4)

    return data, affine


class Visualizer:

    def montage_nrrd(self, image):
        if len(image.shape) > 2:
            return montage(image)
        else:
            warnings.warn('Pass a 3D volume', RuntimeWarning)
            return image

    def visualize(self, image, mask=None, path_save=None):

        if mask is None:
            fig, axes = plt.subplots(1, 1, figsize=(10, 10))
            axes.imshow(self.montage_nrrd(image))
            axes.set_axis_off()
        else:
            fig, axes = plt.subplots(1, 2, figsize=(40, 40))

            for i, data in enumerate([image, mask]):
                axes[i].imshow(self.montage_nrrd(data))
                axes[i].set_axis_off()
                plt.savefig(path_save)
            plt.close(fig)


def get_subjects(file_spect_data, folder_volumes):

    path_data = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        'data'
    )
    path_volumes = os.path.join(path_data, folder_volumes)
    df = pd.read_csv(os.path.join(path_data, 'eda', file_spect_data))
    subjects = []
    segmentations = list(
        df.loc[(df['mask'].isnull()), ['image']].to_records(index=False)
    )
    segmentations = list(map(
        lambda x: os.path.join(path_volumes, x[0]),
        segmentations
    ))
    for image_path in segmentations:
        subject = tio.Subject(
            spect=tio.ScalarImage(image_path, reader=load_image)
        )
        subjects.append(subject)

    print(f"Dataset size: {len(subjects)} subjects")

    return subjects


def get_train_valid_data_loaders(
    subjects,
    training_split_ratio,
    training_transform,
    validation_transform,
    training_batch_size,
    validation_batch_size,
    num_workers=0
):
    num_subjects = len(subjects)
    num_training_subjects = int(training_split_ratio * num_subjects)
    num_validation_subjects = num_subjects - num_training_subjects

    num_split_subjects = (num_training_subjects, num_validation_subjects)
    training_subjects, validation_subjects = torch.utils.data.random_split(
        subjects, num_split_subjects
    )
    training_set = tio.SubjectsDataset(
        training_subjects,
        transform=training_transform
    )
    validation_set = tio.SubjectsDataset(
        validation_subjects,
        transform=validation_transform
    )
    print('Training set:', len(training_set), 'subjects')
    print('Validation set:', len(validation_set), 'subjects')

    training_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=training_batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=validation_batch_size,
        num_workers=num_workers
    )
    return training_loader, validation_loader


def replace_target_norm_layer_for_group_norm(
    model,
    target,
    desired,
    num_groups=8
):
    """

    :param model:
    :param target:
    :param desired:
    :param num_groups:
    :return:
    """
    print(
        f"Replacing {target.__name__} layers for {desired.__name__} layers..."
    )
    for child_name, child in model.named_children():
        if isinstance(child, target):
            setattr(
                model,
                child_name,
                desired(num_groups=num_groups, num_channels=child.num_features)
            )
        else:
            replace_target_norm_layer_for_group_norm(
                child,
                target,
                desired,
                num_groups
            )


def get_annotated_subjects(file_spect_data, folder_volumes):
    """

    :param file_spect_data:
    :param folder_volumes:
    :return:
    """
    path_data = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        'data'
    )
    path_volumes = os.path.join(path_data, folder_volumes)

    df = pd.read_csv(os.path.join(path_data, 'eda', file_spect_data))
    subjects = []

    segmentations = list(
        df.loc[
            (df['mask'].notnull()),
            ['image', 'mask']
        ].to_records(index=False)
    )
    segmentations = list(map(
        lambda x: (
            os.path.join(path_volumes, x[0]),
            os.path.join(path_volumes, x[1])
        ),
        segmentations
    ))
    for image_path, label_path in segmentations:
        subject = tio.Subject(
            spect=tio.ScalarImage(image_path, reader=load_image),
            left_ventricle=tio.LabelMap(label_path, reader=load_image)
        )
        subjects.append(subject)

    print(f"There are {len(subjects)} annotated subjects")

    return subjects
