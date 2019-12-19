from keras.layers import Input, Lambda, concatenate
from keras.models import Model


def DCGAN(generator, discriminator, input_dim, patch_size, mini_patch=False):
    generator_input = Input(batch_shape=input_dim, name="DCGAN_input")
    generated_image = generator(generator_input)

    if mini_patch:

        x, y, z = input_dim[1:4]
        px, py, pz = patch_size[:]
        list_z_idx = [(i * pz, (i + 1) * pz) for i in range(int(z / pz))]
        list_y_idx = [(i * py, (i + 1) * py) for i in range(int(y / py))]
        list_x_idx = [(i * px, (i + 1) * px) for i in range(int(x / px))]

        image_patches = []
        for x_idx in list_x_idx:
            for y_idx in list_y_idx:
                for z_idx in list_z_idx:
                    patch = Lambda(lambda z: z[:, x_idx[0]:x_idx[1], y_idx[0]:y_idx[1], z_idx[0]:z_idx[1], :])(generated_image)
                    image_patches.append(patch)

        image_patches = concatenate(image_patches,axis=0)

        dcgan_output = discriminator(image_patches)
        dc_gan = Model(input=[generator_input], output=[generated_image, dcgan_output], name="DCGAN")

    else:

        dcgan_output = discriminator(generated_image)
        dc_gan = Model(input=[generator_input], output=[generated_image, dcgan_output], name="DCGAN")

    return dc_gan

