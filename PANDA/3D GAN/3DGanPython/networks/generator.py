from keras.layers.merge import concatenate
import numpy as np
from keras import backend as K
from keras.initializers import RandomNormal, Zeros
from keras.engine import Input, Model
from keras.layers.convolutional import Conv3D, Deconv3D, Conv3DTranspose
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Deconvolution3D, ReLU, SpatialDropout3D
from keras.layers.advanced_activations import LeakyReLU

''' Generator structure based on Unet. The activation function of the last layer is relu. In case you normalize the data in a different way
or you want to use the script for another dataset type, please revise the activaction function of the generator. If you wish to apply any modification to the generator, you can modify 
the code below. No modification can be apply from the main command line'''


def create_convolution_block(input_layer, n_filters, batch_normalization=True, kernel_size=(4, 4, 4), activation=LeakyReLU(alpha=0.2),
                             padding='same', strides=(2, 2, 2), instance_normalization=False):

    init = RandomNormal(mean=0.0, stddev=0.02)
    layer = Conv3D(n_filters, kernel_size, padding=padding, kernel_initializer=init, strides=strides)(input_layer)

    if batch_normalization:
        layer = BatchNormalization( axis=4)(layer)  # channel_last convention

    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=4)(layer)

    layer = Activation(activation)(layer)
    return layer

def bottleneck(input_layer, n_filters, batch_normalization=True, kernel_size=(4, 4, 4), activation='relu',
                             padding='same', strides=(2, 2, 2), instance_normalization=False):

    init = RandomNormal(mean=0.0, stddev=0.02)
    layer = Conv3D(n_filters, kernel_size, padding=padding, kernel_initializer=init, strides=strides)(input_layer)

    if batch_normalization:
        layer = BatchNormalization( axis=4)(layer)

    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=4)(layer)

    layer = Activation(activation)(layer)
    return layer

def create_convolution_block_up(input_layer,skip_conn, n_filters, batch_normalization=True, kernel_size=(4, 4, 4), activation='relu',
                             padding='same', strides=(2, 2, 2), instance_normalization=False, dropout=True):

    init = RandomNormal(mean=0.0, stddev=0.02)
    layer = Conv3DTranspose(n_filters, kernel_size, padding=padding, kernel_initializer=init, strides=strides)(input_layer)

    if batch_normalization:
        layer = BatchNormalization(axis=4)(layer)

    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=4)(layer)

    if dropout:
        layer = SpatialDropout3D(rate=0.5)(layer)

    layer = concatenate([layer, skip_conn], axis=4)

    layer = Activation(activation)(layer)

    return layer


def build_generator(img_shape, gf):
    """U-Net Generator"""

    def conv3d(layer_input, filters, f_size=(4,4,4), bn=True):

        d = create_convolution_block(input_layer=layer_input, n_filters=filters,
                                        batch_normalization=bn, strides=(2, 2, 2), kernel_size=f_size)
        return d

    def deconv3d(layer_input, skip_input, filters, f_size=(4,4,4),drop=True):
        """Layers used during upsampling"""

        u = create_convolution_block_up(input_layer=layer_input,skip_conn=skip_input, n_filters=filters,
                                     batch_normalization=True, strides=(2, 2, 2), kernel_size=f_size, dropout=drop)

        return u

    # Image input
    d0 = Input(batch_shape=img_shape)

    # Downsampling
    e1 = conv3d(d0, gf, bn=False)  # 64
    e2 = conv3d(e1, gf * 2)        # 128
    e3 = conv3d(e2, gf * 4)        # 256
    e4 = conv3d(e3, gf * 8)        # 512
    e5 = conv3d(e4, gf * 8)        # 512

    # bottleneck
    e6 = bottleneck(e5, gf * 8, batch_normalization=False, kernel_size=(4, 4, 4), activation='relu',
                             padding='same', strides=(2, 2, 2), instance_normalization=False)        # 512

    # Upsampling
    u1 = deconv3d(e6, e5, gf * 8, drop=True)
    u2 = deconv3d(u1, e4, gf * 8, drop=True)
    u3 = deconv3d(u2, e3, gf * 4, drop=True)
    u4 = deconv3d(u3, e2, gf * 2, drop=False)
    u5 = deconv3d(u4, e1, gf, drop=False)
    #
    #
    init = RandomNormal(mean=0.0, stddev=0.02)
    u6 = Conv3DTranspose(filters=gf, kernel_size=(4,4,4), padding='same', kernel_initializer=init, strides=(2,2,2))(u5)
    #
    final_convolution = Conv3D(1, (1, 1, 1))(u6)
    act = Activation('relu')(final_convolution)

    return Model(d0, act)


def UNetGenerator(input_dim, filters=64):

    unet_generator = build_generator(input_dim, filters)

    return unet_generator

