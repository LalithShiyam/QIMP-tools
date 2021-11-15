# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from init import Options
from networks import build_net, update_learning_rate, build_UNETR
# from networks import build_net
import logging
import os
import sys
import tempfile
from glob import glob

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import create_test_image_3d, list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (EnsureType, Compose, LoadImaged, AddChanneld, Transpose,Activations,AsDiscrete, RandGaussianSmoothd, CropForegroundd, SpatialPadd,
                              ScaleIntensityd, ToTensord, RandSpatialCropd, Rand3DElasticd, RandAffined, RandZoomd,
    Spacingd, Orientationd, Resized, ThresholdIntensityd, RandShiftIntensityd, BorderPadd, RandGaussianNoised, RandAdjustContrastd,NormalizeIntensityd,RandFlipd)

from monai.visualize import plot_2d_or_3d_image

def main():
    opt = Options().parse()
    # monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # check gpus
    if opt.gpu_ids != '-1':
        num_gpus = len(opt.gpu_ids.split(','))
    else:
        num_gpus = 0
    print('number of GPU:', num_gpus)

    # Data loader creation
    # train images
    train_images = sorted(glob(os.path.join(opt.images_folder, 'train', 'image*.nii')))
    train_segs = sorted(glob(os.path.join(opt.labels_folder, 'train', 'label*.nii')))

    train_images_for_dice = sorted(glob(os.path.join(opt.images_folder, 'train', 'image*.nii')))
    train_segs_for_dice = sorted(glob(os.path.join(opt.labels_folder, 'train', 'label*.nii')))

    # validation images
    val_images = sorted(glob(os.path.join(opt.images_folder, 'val', 'image*.nii')))
    val_segs = sorted(glob(os.path.join(opt.labels_folder, 'val', 'label*.nii')))

    # test images
    test_images = sorted(glob(os.path.join(opt.images_folder, 'test', 'image*.nii')))
    test_segs = sorted(glob(os.path.join(opt.labels_folder, 'test', 'label*.nii')))

    # augment the data list for training
    for i in range(int(opt.increase_factor_data)):
    
        train_images.extend(train_images)
        train_segs.extend(train_segs)

    print('Number of training patches per epoch:', len(train_images))
    print('Number of training images per epoch:', len(train_images_for_dice))
    print('Number of validation images per epoch:', len(val_images))
    print('Number of test images per epoch:', len(test_images))

    # Creation of data directories for data_loader

    train_dicts = [{'image': image_name, 'label': label_name}
                  for image_name, label_name in zip(train_images, train_segs)]

    train_dice_dicts = [{'image': image_name, 'label': label_name}
                   for image_name, label_name in zip(train_images_for_dice, train_segs_for_dice)]

    val_dicts = [{'image': image_name, 'label': label_name}
                   for image_name, label_name in zip(val_images, val_segs)]

    test_dicts = [{'image': image_name, 'label': label_name}
                 for image_name, label_name in zip(test_images, test_segs)]

    # Transforms list

    if opt.resolution is not None:
        train_transforms = [
            LoadImaged(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),
            # ThresholdIntensityd(keys=['image'], threshold=-135, above=True, cval=-135),  # CT HU filter
            # ThresholdIntensityd(keys=['image'], threshold=215, above=False, cval=215),
            CropForegroundd(keys=['image', 'label'], source_key='image'),               # crop CropForeground

            NormalizeIntensityd(keys=['image']),                                          # augmentation
            ScaleIntensityd(keys=['image']),                                              # intensity
            Spacingd(keys=['image', 'label'], pixdim=opt.resolution, mode=('bilinear', 'nearest')),  # resolution

            RandFlipd(keys=['image', 'label'], prob=0.15, spatial_axis=1),
            RandFlipd(keys=['image', 'label'], prob=0.15, spatial_axis=0),
            RandFlipd(keys=['image', 'label'], prob=0.15, spatial_axis=2),
            RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                        rotate_range=(np.pi / 36, np.pi / 36, np.pi * 2), padding_mode="zeros"),
            RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                        rotate_range=(np.pi / 36, np.pi / 2, np.pi / 36), padding_mode="zeros"),
            RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                        rotate_range=(np.pi / 2, np.pi / 36, np.pi / 36), padding_mode="zeros"),
            Rand3DElasticd(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                           sigma_range=(5, 8), magnitude_range=(100, 200), scale_range=(0.15, 0.15, 0.15),
                           padding_mode="zeros"),
            RandGaussianSmoothd(keys=["image"], sigma_x=(0.5, 1.15), sigma_y=(0.5, 1.15), sigma_z=(0.5, 1.15), prob=0.1,),
            RandAdjustContrastd(keys=['image'], gamma=(0.5, 2.5), prob=0.1),
            RandGaussianNoised(keys=['image'], prob=0.1, mean=np.random.uniform(0, 0.5), std=np.random.uniform(0, 15)),
            RandShiftIntensityd(keys=['image'], offsets=np.random.uniform(0,0.3), prob=0.1),

            SpatialPadd(keys=['image', 'label'], spatial_size=opt.patch_size, method= 'end'),  # pad if the image is smaller than patch
            RandSpatialCropd(keys=['image', 'label'], roi_size=opt.patch_size, random_size=False),
            ToTensord(keys=['image', 'label'])
        ]

        val_transforms = [
            LoadImaged(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),
            # ThresholdIntensityd(keys=['image'], threshold=-135, above=True, cval=-135),
            # ThresholdIntensityd(keys=['image'], threshold=215, above=False, cval=215),
            CropForegroundd(keys=['image', 'label'], source_key='image'),                   # crop CropForeground

            NormalizeIntensityd(keys=['image']),                                      # intensity
            ScaleIntensityd(keys=['image']),
            Spacingd(keys=['image', 'label'], pixdim=opt.resolution, mode=('bilinear', 'nearest')),  # resolution

            SpatialPadd(keys=['image', 'label'], spatial_size=opt.patch_size, method= 'end'),  # pad if the image is smaller than patch
            ToTensord(keys=['image', 'label'])
        ]
    else:
        train_transforms = [
            LoadImaged(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),
            # ThresholdIntensityd(keys=['image'], threshold=-135, above=True, cval=-135),
            # ThresholdIntensityd(keys=['image'], threshold=215, above=False, cval=215),
            CropForegroundd(keys=['image', 'label'], source_key='image'),               # crop CropForeground

            NormalizeIntensityd(keys=['image']),                                          # augmentation
            ScaleIntensityd(keys=['image']),                                              # intensity

            RandFlipd(keys=['image', 'label'], prob=0.15, spatial_axis=1),
            RandFlipd(keys=['image', 'label'], prob=0.15, spatial_axis=0),
            RandFlipd(keys=['image', 'label'], prob=0.15, spatial_axis=2),
            RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                        rotate_range=(np.pi / 36, np.pi / 36, np.pi * 2), padding_mode="zeros"),
            RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                        rotate_range=(np.pi / 36, np.pi / 2, np.pi / 36), padding_mode="zeros"),
            RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                        rotate_range=(np.pi / 2, np.pi / 36, np.pi / 36), padding_mode="zeros"),
            Rand3DElasticd(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                           sigma_range=(5, 8), magnitude_range=(100, 200), scale_range=(0.15, 0.15, 0.15),
                           padding_mode="zeros"),
            RandGaussianSmoothd(keys=["image"], sigma_x=(0.5, 1.15), sigma_y=(0.5, 1.15), sigma_z=(0.5, 1.15), prob=0.1,),
            RandAdjustContrastd(keys=['image'], gamma=(0.5, 2.5), prob=0.1),
            RandGaussianNoised(keys=['image'], prob=0.1, mean=np.random.uniform(0, 0.5), std=np.random.uniform(0, 1)),
            RandShiftIntensityd(keys=['image'], offsets=np.random.uniform(0,0.3), prob=0.1),

            SpatialPadd(keys=['image', 'label'], spatial_size=opt.patch_size, method= 'end'),  # pad if the image is smaller than patch
            RandSpatialCropd(keys=['image', 'label'], roi_size=opt.patch_size, random_size=False),
            ToTensord(keys=['image', 'label'])
        ]

        val_transforms = [
            LoadImaged(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),
            # ThresholdIntensityd(keys=['image'], threshold=-135, above=True, cval=-135),
            # ThresholdIntensityd(keys=['image'], threshold=215, above=False, cval=215),
            CropForegroundd(keys=['image', 'label'], source_key='image'),                   # crop CropForeground

            NormalizeIntensityd(keys=['image']),                                      # intensity
            ScaleIntensityd(keys=['image']),

            SpatialPadd(keys=['image', 'label'], spatial_size=opt.patch_size, method= 'end'),  # pad if the image is smaller than patch
            ToTensord(keys=['image', 'label'])
        ]

    train_transforms = Compose(train_transforms)
    val_transforms = Compose(val_transforms)

    # create a training data loader
    check_train = monai.data.Dataset(data=train_dicts, transform=train_transforms)
    train_loader = DataLoader(check_train, batch_size=opt.batch_size, shuffle=True, collate_fn=list_data_collate, num_workers=opt.workers, pin_memory=False)

    # create a training_dice data loader
    check_val = monai.data.Dataset(data=train_dice_dicts, transform=val_transforms)
    train_dice_loader = DataLoader(check_val, batch_size=1, num_workers=opt.workers, collate_fn=list_data_collate, pin_memory=False)

    # create a validation data loader
    check_val = monai.data.Dataset(data=val_dicts, transform=val_transforms)
    val_loader = DataLoader(check_val, batch_size=1, num_workers=opt.workers, collate_fn=list_data_collate, pin_memory=False)

    # create a validation data loader
    check_val = monai.data.Dataset(data=test_dicts, transform=val_transforms)
    test_loader = DataLoader(check_val, batch_size=1, num_workers=opt.workers, collate_fn=list_data_collate, pin_memory=False)

    # build the network
    if opt.network is 'nnunet':
        net = build_net()  # nn build_net
    elif opt.network is 'unetr':
        net = build_UNETR() # UneTR
    net.cuda()

    if num_gpus > 1:
        net = torch.nn.DataParallel(net)

    if opt.preload is not None:
        net.load_state_dict(torch.load(opt.preload))

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)])

    loss_function = monai.losses.DiceCELoss(sigmoid=True)
    torch.backends.cudnn.benchmark = opt.benchmark


    if opt.network is 'nnunet':

        optim = torch.optim.SGD(net.parameters(), lr=opt.lr, momentum=0.99, weight_decay=3e-5, nesterov=True,)
        net_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda epoch: (1 - epoch / opt.epochs) ** 0.9)

    elif opt.network is 'unetr':

        optim = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-5)

    # start a typical PyTorch training
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    writer = SummaryWriter()
    for epoch in range(opt.epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{opt.epochs}")
        net.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].cuda(), batch_data["label"].cuda()
            optim.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            epoch_len = len(check_train) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        if opt.network is 'nnunet':
            update_learning_rate(net_scheduler, optim)

        if (epoch + 1) % val_interval == 0:
            net.eval()
            with torch.no_grad():

                def plot_dice(images_loader):

                    val_images = None
                    val_labels = None
                    val_outputs = None
                    for data in images_loader:
                        val_images, val_labels = data["image"].cuda(), data["label"].cuda()
                        roi_size = opt.patch_size
                        sw_batch_size = 4
                        val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, net)
                        val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                        dice_metric(y_pred=val_outputs, y=val_labels)

                    # aggregate the final mean dice result
                    metric = dice_metric.aggregate().item()
                    # reset the status for next validation round
                    dice_metric.reset()

                    return metric, val_images, val_labels, val_outputs

                metric, val_images, val_labels, val_outputs = plot_dice(val_loader)

                # Save best model
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(net.state_dict(), "best_metric_model.pth")
                    print("saved new best metric model")

                metric_train, train_images, train_labels, train_outputs = plot_dice(train_dice_loader)
                metric_test, test_images, test_labels, test_outputs = plot_dice(test_loader)

                # Logger bar
                print(
                    "current epoch: {} Training dice: {:.4f} Validation dice: {:.4f} Testing dice: {:.4f} Best Validation dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric_train, metric, metric_test, best_metric, best_metric_epoch
                    )
                )

                writer.add_scalar("Mean_epoch_loss", epoch_loss, epoch + 1)
                writer.add_scalar("Testing_dice", metric_test, epoch + 1)
                writer.add_scalar("Training_dice", metric_train, epoch + 1)
                writer.add_scalar("Validation_dice", metric, epoch + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                # val_outputs = (val_outputs.sigmoid() >= 0.5).float()
                plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="validation image")
                plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="validation label")
                plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="validation inference")
                plot_2d_or_3d_image(test_images, epoch + 1, writer, index=0, tag="test image")
                plot_2d_or_3d_image(test_labels, epoch + 1, writer, index=0, tag="test label")
                plot_2d_or_3d_image(test_outputs, epoch + 1, writer, index=0, tag="test inference")

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


if __name__ == "__main__":
    main()
