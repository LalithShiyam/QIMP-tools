from keras.layers.merge import concatenate
import numpy as np
from keras import backend as K
from keras.initializers import RandomNormal, Zeros
from keras.engine import Input, Model
from keras.layers.convolutional import Conv3D, Deconv3D, Conv3DTranspose
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Deconvolution3D, ReLU, SpatialDropout3D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam


from typing import Tuple, Union
import keras
from keras import backend as K
from keras.layers import Layer
import numpy as np

class BlurPool3D(Layer):
    """
        https://arxiv.org/abs/1904.11486
        Keras implementation of BlurPool3D layer
         for "channels_last" image data format
        Original 1D and 2D PyTorch implementation can be found at
        https://github.com/adobe/antialiased-cnns
    """

    def __init__(
        self,
        pool_size: Union[int, Tuple[int, int, int]],
        kernel_size: Union[int, Tuple[int, int, int]],
        
    ):
        if isinstance(pool_size, int):
            self.pool_size = (pool_size,) * 3
        else:
            self.pool_size = pool_size

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size,) * 3
        else:
            self.kernel_size = kernel_size

        self.blur_kernel = None

        self.padding = tuple(
            (int(1.0 * (size - 1) / 2), int(np.ceil(1.0 * (size - 1) / 2)))
            for size in self.kernel_size
        )

        super().__init__()

    def build(self, input_shape):

        kernel_to_array = {
            1: np.array([1.0]),
            2: np.array([1.0, 1.0]),
            3: np.array([1.0, 2.0, 1.0]),
            4: np.array([1.0, 3.0, 3.0, 1.0]),
            5: np.array([1.0, 4.0, 6.0, 4.0, 1.0]),
            6: np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0]),
            7: np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0]),
        }

        a = kernel_to_array[self.kernel_size[0]]
        b = kernel_to_array[self.kernel_size[1]]
        c = kernel_to_array[self.kernel_size[2]]

        bk = a[:, None, None] * b[None, :, None] * c[None, None, :]
        bk = bk / np.sum(bk)
        bk = np.repeat(bk, input_shape[4])

        new_shape = (*self.kernel_size, input_shape[4], 1)
        bk = np.reshape(bk, new_shape)
        blur_init = keras.initializers.constant(bk)

        self.blur_kernel = self.add_weight(
            name='blur_kernel',
            shape=new_shape,
            initializer=blur_init,
            trainable=False,
        )

        super().build(input_shape)

    def call(self, x, **kwargs):
        x = K.spatial_3d_padding(x, padding=self.padding)

        # we imitate depthwise_conv3d actually
        channels = x.shape[-1]
        x = K.concatenate(
            [
                K.conv3d(
                    x=x[:, :, :, :, i : i + 1],
                    kernel=self.blur_kernel[..., i : i + 1, :],
                    strides=self.pool_size,
                    padding='valid',
                )
                for i in range(0, channels)
            ],
            axis=-1,
        )

        return x

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            int(np.ceil(input_shape[1] / self.pool_size[0])),
            int(np.ceil(input_shape[2] / self.pool_size[1])),
            int(np.ceil(input_shape[3] / self.pool_size[2])),
            input_shape[4],
        )

    def get_config(self):
        base_config = super().get_config()
        base_config['pool_size'] = self.pool_size
        base_config['kernel_size'] = self.kernel_size

        return base_config

def create_convolution_block(input_layer, n_filters, batch_normalization=True, kernel_size=(4, 4, 4), activation=LeakyReLU(alpha=0.2),
                             padding='same', strides=(2, 2, 2), instance_normalization=False):

    # 3DConv + Normalization + Activation
    # Instance Normalization is said to perform better than Batch Normalization

    init = RandomNormal(mean=0.0, stddev=0.02) # new
    layer = Conv3D(n_filters, kernel_size, padding=padding, kernel_initializer=init, strides=strides)(input_layer)

    if batch_normalization:
        layer = BatchNormalization(axis=4)(layer)  # channel_last convention
    # elif instance_normalization:
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

    # 3DConv + Normalization + Activation
    # Instance Normalization is said to perform better than Batch Normalization

    init = RandomNormal(mean=0.0, stddev=0.02) # new
    layer = Conv3D(n_filters, kernel_size, padding=padding, kernel_initializer=init, strides=strides)(input_layer)

    if batch_normalization:
        layer = BatchNormalization( axis=4)(layer)  # channel_last convention
    # elif instance_normalization:
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

    # 3DConv + Normalization + Activation
    # Instance Normalization is said to perform better than Batch Normalization

    init = RandomNormal(mean=0.0, stddev=0.02)  # new
    layer = Conv3DTranspose(n_filters, kernel_size, padding=padding, kernel_initializer=init, strides=strides)(input_layer)

    if batch_normalization:
        layer = BatchNormalization(axis=4)(layer)  # channel_last convention
    # elif instance_normalization:
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
    init = RandomNormal(mean=0.0, stddev=0.02)  # new
    u6 = Conv3DTranspose(filters=gf, kernel_size=(4,4,4), padding='same', kernel_initializer=init, strides=(2,2,2))(u5)
    #
    final_convolution = Conv3D(1, (1, 1, 1))(u6)
    act = Activation('relu')(final_convolution)

    return Model(d0, act)


def UNetGenerator(input_dim, filters=64):

    unet_generator = build_generator(input_dim , filters)

    return unet_generator

