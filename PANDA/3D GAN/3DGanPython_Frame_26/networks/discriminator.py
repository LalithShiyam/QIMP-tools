import numpy as np
from keras.initializers import RandomNormal
from keras.engine import Input, Model
from keras.layers import Conv3D, Flatten, Dense, BatchNormalization, SpatialDropout3D,ZeroPadding3D
# from keras.layers.merge import concatenate
# from keras_contrib.layers.normalization import InstanceNormalization
# from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers.advanced_activations import LeakyReLU
import math
from keras.initializers import RandomNormal, Zeros
from predict import prepare_batch
import random


def smooth_positive_labels(y):
    output = y - 0.3 + (np.random.random(y) * 0.5)

    if output >= 1:

        return 1
    else:
        return output

def smooth_negative_labels(y):
    return 0 + np.random.random(y) * 0.3



def get_patches(source_images_for_training, target_images_for_training, mini_patch_size, generator, batch_counter, smooth_labels=True):

    input_dim = source_images_for_training.shape
    z, y, x = input_dim[1:4]
    pz, py, px = mini_patch_size[:]

    list_z_idx = [(i * pz, (i + 1) * pz) for i in range(int(z / pz))]
    list_y_idx = [(i * py, (i + 1) * py) for i in range(int(y / py))]
    list_x_idx = [(i * px, (i + 1) * px) for i in range(int(x / px))]

    image = np.zeros(input_dim)
    patch_images = []
    for z_idx in list_z_idx:
        for y_idx in list_y_idx:
            for x_idx in list_x_idx:
                patch = image[:, z_idx[0]:z_idx[1], y_idx[0]:y_idx[1], x_idx[0]:x_idx[1], :]
                patch_images.append(patch)

    patch_images = np.asarray(patch_images)
    patch_images = np.squeeze(patch_images, axis=1)
    discriminator_nub_patches = patch_images.shape[0]

    # generate fake image for di batch_counter
    if batch_counter % 2 == 0:

        image = generator.predict(source_images_for_training)
        patch_labels = np.zeros((discriminator_nub_patches,
                                 2))  # you need to change 5 to another number , if the number of subpathes is different

        if smooth_labels:
            patch_labels[:, 0] = smooth_positive_labels(1)  # set the first column to 1 because they are fake image
            patch_labels[:, 1] = smooth_negative_labels(1)  # set the first column to 1 because they are fake image
            if random.randint(0, 100) <= 5:
                patch_labels[:, 0] = smooth_negative_labels(1)  # set the first column to 1 because they are fake image
                patch_labels[:, 1] = smooth_positive_labels(1)  # set the first column to 1 because they are fake image

        else:
            patch_labels[:, 0] = 1  # set the first column to 1 because they are fake image
            if random.randint(0, 100) <= 5:
                patch_labels[:, 1] = 1  # Use Noisy Labels

    else:
        image = target_images_for_training
        patch_labels = np.zeros((discriminator_nub_patches, 2))

        if smooth_labels:
            patch_labels[:, 0] = smooth_negative_labels(1)  # set the first column to 1 because they are fake image
            patch_labels[:, 1] = smooth_positive_labels(1)  # set the first column to 1 because they are fake image
            if random.randint(0, 100) <= 5:
                patch_labels[:, 0] = smooth_positive_labels(1)  # set the first column to 1 because they are fake image
                patch_labels[:, 1] = smooth_negative_labels(1)  # set the first column to 1 because they are fake image

        else:
            patch_labels[:, 1] = 1  # set the first column to 1 because they are real image
            if random.randint(0, 100) <= 5:
                patch_labels[:, 0] = 1  # Use Noisy Labels

    patch_images = []
    for z_idx in list_z_idx:
        for y_idx in list_y_idx:
            for x_idx in list_x_idx:
                patch = image[:, z_idx[0]:z_idx[1], y_idx[0]:y_idx[1], x_idx[0]:x_idx[1], :]
                patch_images.append(patch)

    patch_images=np.asarray(patch_images)
    patch_images=np.squeeze(patch_images, axis=1)

    return patch_images, patch_labels


def PatchGanDiscriminator(output_dim, patch_size, padding='same', strides=(2,2,2), kernel_size=(4,4,4), batch_norm=True, dropout= True):

    inputs = Input(shape=[patch_size[0], patch_size[1], patch_size[2], output_dim[4]])
    filter_list = [64, 128, 256, 512, 512, 512]

    # Layer1 without Batch Normalization

    disc_out = Conv3D(filters=filter_list[0], kernel_size=kernel_size,kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                                   bias_initializer=Zeros(), padding=padding, strides=strides)(inputs)
    disc_out = LeakyReLU(alpha=0.2)(disc_out)
    # disc_out = BatchNormalization(axis=4)(disc_out)  # Original one with Batch normalization


    # build the rest Layers
    # Conv -> BN -> LeakyReLU
    for i, filter_size in enumerate(filter_list[1:]):
        name = 'disc_conv_{}'.format(i+1)


        disc_out = Conv3D(name=name, filters=filter_list[i+1],kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                                   bias_initializer=Zeros(), kernel_size=kernel_size, padding=padding, strides=strides)(disc_out)

        if batch_norm:
            disc_out = BatchNormalization(axis=4)(disc_out)  # channel_last convention
        if dropout:
            disc_out = SpatialDropout3D(rate=0.5)(disc_out)

        disc_out = LeakyReLU(alpha=0.2)(disc_out)


    x_flat = Flatten()(disc_out)
    x = Dense(2, activation='sigmoid',name="disc_dense")(x_flat)
    patch_GAN_discriminator = Model(input=inputs, output=x, name="patch_gan")

    return patch_GAN_discriminator

