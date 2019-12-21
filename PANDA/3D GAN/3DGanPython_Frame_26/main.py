from __future__ import division, print_function
from functools import partial
from keras import backend as K
from utils.data_generator import *
from logger import *
from networks.generator import UNetGenerator
from networks.discriminator import PatchGanDiscriminator, get_patches
from networks.DCGAN import DCGAN
from keras.optimizers import Adam
from keras.utils import generic_utils as keras_generic_utils
from utils.NiftiDataset import *
import utils.NiftiDataset as NiftiDataset
import os
import time
import numpy as np
import tensorflow as tf
from predict import *
import argparse

''' 

The network was realized to generate high-dose Pet images from low-dose frames.

The main script is divided in 3 sequential branches: creation of training and testing dataset, training and validation of the model. The use can decide to execute all of them or to run only
some of them by setting to True/False the respective options from the command line. Some parameters are set by default to a certain value. Please modify this value or digit from the command
line the parameter's values you're interested to change.

The script runs the training of a 3D GAN based on pix2pix. (Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio.
Generative Adversarial Networks. NIPS, 2014)

The generator can take as input the entire volume or patches of the volume. The patches will reconstruct the original volume after the inference is run. 
The discriminator can take as input the output of the generator or its sub-patches. The user has to choose this option.  


'''

if __name__ == "__main__":

    # ---------------------------------------
    # Set up the params from the command line
    # ---------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument("--Use_GPU", action='store_true', default=True, help='Use the GPU')
    parser.add_argument("--Select_GPU", type=int, default=1, help='Select the GPU')
    parser.add_argument("--Create_training_test_dataset", action='store_true', default=True, help='Divide the data for the training')
    parser.add_argument("--Do_you_wanna_train", action='store_true', default=True, help='Training will start')
    parser.add_argument("--Do_you_wanna_load_weights", action='store_true', default=False, help='Load weights')
    parser.add_argument("--Do_you_wanna_check_accuracy", action='store_true', default=False, help='Model will be tested after the training')
    parser.add_argument("--save_dir", type=str, default='./Data_folder/', help='path to folders with low dose and high dose folders')
    parser.add_argument("--images_folder", type=str, default='./Data_folder/volumes', help='path to the .nii low dose images')
    parser.add_argument("--labels_folder", type=str, default='./Data_folder/labels', help='path to the .nii high dose images')
    parser.add_argument("--val_split", type=float, default=0.1, help='Split value for the validation data (0 to 1 float number)')
    parser.add_argument("--history_dir", type=str, default='./History', help='path where to save sample images during training')
    parser.add_argument("--weights", type=str, default='./History/weights', help='path to save the weights of the model')
    parser.add_argument("--gen_weights", type=str, default='./History/weights/gen_weights_frame_26.h5', help='generator weights to load')
    parser.add_argument("--disc_weights", type=str, default='./History/weights/disc_weights_epoch_20.h5', help='generator weights to load')
    parser.add_argument("--dcgan_weights", type=str, default='./History/weights/DCGAN_weights_epoch_20.h5', help='generator weights to load')

    # Training parameters
    parser.add_argument("--resample", action='store_true', default=False, help='Decide or not to resample the images to a new resolution')
    parser.add_argument("--new_resolution", type=float, default=(1.5, 1.5, 1.5), help='New resolution')
    parser.add_argument("--input_channels", type=float, nargs=1, default=1, help="Input channels")
    parser.add_argument("--output_channels", type=float, nargs=1, default=1, help="Output channels (Current implementation supports one output channel")
    parser.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 64], help="Input dimension for the generator")
    parser.add_argument("--mini_patch", action='store_true', default=True, help=' If True, discriminator and DCgan will be trained with subpatches of the generator input')
    parser.add_argument("--mini_patch_size", type=int, nargs=3, default=[64, 64, 64], help="Input dimension for the discriminator and DCgan")
    parser.add_argument("--batch_size", type=int, nargs=1, default=1, help="Batch size to feed the network (currently supports 1)")
    parser.add_argument("--drop_ratio", type=float, nargs=1, default=0, help="Probability to drop a cropped area if the label is empty. All empty patches will be dropped for 0 and accept all cropped patches if set to 1")
    parser.add_argument("--min_pixel", type=int, nargs=1, default=1, help="Percentage of minimum non-zero pixels in the cropped label")

    parser.add_argument("--lr", type=float, nargs=1, default=0.0002, help="learning rate")
    parser.add_argument("--beta_1", type=float, nargs=1, default=0.5, help="beta 1")
    parser.add_argument("--beta_2", type=float, nargs=1, default=0.999, help="beta 2")
    parser.add_argument("--epsilon", type=float, nargs=1, default=1e-8, help="epsilon optimizer")
    parser.add_argument("--nb_epoch", type=int, nargs=1, default=200, help="number of epochs")
    parser.add_argument("--n_images_per_epoch", type=int, nargs=1, default=400, help="Number of images per epoch")
    # Inference parameters
    parser.add_argument("--stride_inplane", type=int, nargs=1, default=16, help="Stride size in 2D plane")
    parser.add_argument("--stride_layer", type=int, nargs=1, default=16, help="Stride size in z direction")
    args = parser.parse_args()

    if args.Use_GPU is True:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.Select_GPU)

    if not os.path.exists(args.history_dir):
        os.makedirs(args.history_dir)
    if not os.path.exists(args.weights):
        os.makedirs(args.weights)

    min_pixel = int(args.min_pixel*((args.patch_size[0]*args.patch_size[1]*args.patch_size[2])/100))

    # ---------------------------------------
    # 1) Create training and validation lists
    # ---------------------------------------

    images = lstFiles(args.images_folder)
    labels = lstFiles(args.labels_folder)

    if args.Create_training_test_dataset is True:

        images, val_images, labels, val_labels = split_train_set(args.val_split, images, labels)

        write_list(args.save_dir + '/' + 'train.txt', images)
        write_list(args.save_dir + '/' + 'train_labels.txt', labels)
        write_list(args.save_dir + '/' + 'val.txt', val_images)
        write_list(args.save_dir + '/' + 'val_labels.txt', val_labels)

    # -----------------
    # 2) Training model
    # -----------------

    if args.Do_you_wanna_train is True:

        f = open(args.save_dir + '/' + 'train_labels.txt', 'r')
        n_samples_train = len(f.readlines())
        f.close()

        f = open(args.save_dir + '/' + 'val.txt', 'r')
        n_samples_val = len(f.readlines())
        f.close()

        print('Number of training samples:', n_samples_train, '  Number of validation samples:', n_samples_val)

        trainTransforms = [
            # NiftiDataset.StatisticalNormalization(2.5),
            # NiftiDataset.Normalization(),
            NiftiDataset.Resample(args.new_resolution, args.resample),
            NiftiDataset.Augmentation(),
            NiftiDataset.Padding((args.patch_size[0], args.patch_size[1], args.patch_size[2])),
            NiftiDataset.RandomCrop((args.patch_size[0], args.patch_size[1], args.patch_size[2]),
                                    args.drop_ratio, min_pixel)
        ]

        # data generator
        training_generator = data_generator(images_list=args.save_dir + '/' + 'train.txt', labels_list=args.save_dir + '/' + 'train_labels.txt',
                         batch_size=args.batch_size, Transforms=trainTransforms)

        # input & output dims
        input_dim = [args.batch_size,  args.patch_size[0],  args.patch_size[1], args.patch_size[2], args.input_channels]
        output_dim = [args.batch_size,  args.patch_size[0],  args.patch_size[1], args.patch_size[2], args.output_channels]

        # ------------------
        # Build the networks
        # ------------------

        # optimizers
        opt_generator = Adam(lr=args.lr, beta_1=args.beta_1, beta_2=args.beta_2, epsilon=args.epsilon)
        opt_discriminator = Adam(lr=args.lr, beta_1=args.beta_1, beta_2=args.beta_2, epsilon=args.epsilon)
        opt_DCGAN = Adam(lr=args.lr, beta_1=args.beta_1, beta_2=args.beta_2, epsilon=args.epsilon)

        # UNetGenerator
        generator = UNetGenerator(input_dim=input_dim)
        generator.summary()
        if args.Do_you_wanna_load_weights is True:
            generator.load_weights(args.gen_weights)
        generator.compile(loss='mae', optimizer=opt_generator)

        # Patch GAN Discriminator
        discriminator = PatchGanDiscriminator(output_dim=output_dim, patch_size=args.mini_patch_size)
        discriminator.summary()
        if args.Do_you_wanna_load_weights is True:
            discriminator.load_weights(args.disc_weights)
        discriminator.trainable = False

        # DCGAN
        dc_gan = DCGAN(generator=generator, discriminator=discriminator, input_dim=input_dim, patch_size=args.mini_patch_size)
        if args.Do_you_wanna_load_weights is True:
            discriminator.load_weights(args.dcgan_weights)
        dc_gan.summary()

        # Total Loss
        loss = ['mae', 'binary_crossentropy']
        loss_weights = [1e2, 1]
        dc_gan.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_DCGAN)

        discriminator.trainable = True
        discriminator.compile(loss='binary_crossentropy', optimizer=opt_discriminator)

        # -----------
        # Train cycle
        # -----------

        print("Start training..")
        epoch_count = 1

        D_log_loss_list = []
        gan_total_loss_list = []
        gan_mae_list = []
        gan_log_loss_list = []

        for epoch in range(0, args.nb_epoch):
            print('Epoch {}'.format(epoch))
            batch_counter = 1
            start = time.time()
            progress_bar = keras_generic_utils.Progbar(args.n_images_per_epoch)
            for batch_i in range(0, args.n_images_per_epoch, args.batch_size):

                source_images_for_training, target_images_for_training = next(training_generator)

                patch_images, patch_labels = get_patches(source_images_for_training, target_images_for_training, args.mini_patch_size, generator, batch_counter)

                discriminator_loss = discriminator.train_on_batch(patch_images, patch_labels)
                discriminator.trainable = False

                gan_labels = np.zeros((target_images_for_training.shape[0], 2), dtype=np.uint8)
                gan_labels[:, 1] = 1
                gan_loss = dc_gan.train_on_batch(source_images_for_training, [target_images_for_training, gan_labels])

                discriminator.trainable = True

                batch_counter += 1

                D_log_loss = discriminator_loss
                gan_total_loss = gan_loss[0].tolist()
                gan_total_loss = min(gan_total_loss, 1000000)
                gan_mae = gan_loss[1].tolist()
                gan_mae = min(gan_mae, 1000000)
                gan_log_loss = gan_loss[2].tolist()
                gan_log_loss = min(gan_log_loss, 1000000)

                progress_bar.add(args.batch_size, values=[("Dis logloss", D_log_loss),
                                                ("GAN total", gan_total_loss),
                                                ("GAN L1", gan_mae),
                                                ("GAN logloss", gan_log_loss)])

            plot_generated_batch(image=args.save_dir + '/' + 'val.txt', label=args.save_dir + '/' + 'val_labels.txt',model=generator, resample=args.resample, resolution=args.new_resolution, patch_size_x=args.patch_size[0],
                                 patch_size_y=args.patch_size[1], patch_size_z=args.patch_size[2],stride_inplane=args.stride_inplane, stride_layer=args.stride_layer, batch_size=1,
                                 epoch=epoch_count)

            epoch_count += 1

            # save weights and images on every 10 epoch
            if epoch % 10 == 0:
                gan_weights_path = os.path.join('./History/weights/gen_weights_epoch_%s.h5' % epoch)
                generator.save_weights(gan_weights_path, overwrite=True)

                disc_weights_path = os.path.join('./History/weights/disc_weights_epoch_%s.h5' % epoch)
                discriminator.save_weights(disc_weights_path, overwrite=True)

                DCGAN_weights_path = os.path.join('./History/weights/DCGAN_weights_epoch_%s.h5' % epoch)
                dc_gan.save_weights(DCGAN_weights_path, overwrite=True)

            training_list = ["Dis logloss", D_log_loss_list, "GAN total", gan_total_loss_list, "GAN L1", gan_mae_list,
                             "GAN logloss", gan_log_loss_list]

    # -------------------------------------------------------
    # 3) Check accuracy model on training and validation data
    # -------------------------------------------------------

    if args.Do_you_wanna_check_accuracy is True:

        input_dim = [args.batch_size, args.patch_size[0], args.patch_size[1], args.patch_size[2], args.input_channels]
        model = UNetGenerator(input_dim=input_dim)
        model.load_weights(args.gen_weights)

        check_accuracy_model(model, images_list=args.save_dir + '/' + 'val.txt', labels_list=args.save_dir + '/' + 'val_labels.txt', resample=args.resample,
                             new_resolution=args.new_resolution, patch_size_x=args.patch_size[0],patch_size_y=args.patch_size[1], patch_size_z=args.patch_size[2],
                             stride_inplane=args.stride_inplane, stride_layer=args.stride_layer, batch_size=1)

        check_accuracy_model(model, images_list=args.save_dir + '/' + 'train.txt',
                             labels_list=args.save_dir + '/' + 'train_labels.txt', resample=args.resample,
                             new_resolution=args.new_resolution, patch_size_x=args.patch_size[0],
                             patch_size_y=args.patch_size[1], patch_size_z=args.patch_size[2],
                             stride_inplane=args.stride_inplane, stride_layer=args.stride_layer, batch_size=1)













