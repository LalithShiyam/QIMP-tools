#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from utils.metrics import *
from utils.NiftiDataset import *
from utils.metrics import *
import utils.NiftiDataset as NiftiDataset
from tqdm import tqdm
import datetime
from networks.generator import *
import argparse

''' The script run the inference on the single low dose image chosen by the user. Normalization is performed and images are scaled to interval values: 0-255.
    The path of the input image and the path to save the result must be specified in the command line '''

parser = argparse.ArgumentParser()
parser.add_argument('--Use_GPU', action='store_true', default=True, help='Use the GPU')
parser.add_argument('--Select_GPU', type=int, default=1, help='Select the GPU')
parser.add_argument("--image", type=str, default='./Data_folder/volumes', help='path to the .nii low dose image')
parser.add_argument("--result", type=str, default='./Data_folder/volumes', help='path to the .nii result to save')
parser.add_argument("--gen_weights", type=str, default='./History/weights/gen_weights_epoch_20.h5', help='generator weights to load')
# Training parameters
parser.add_argument("--resample", action='store_true', default=False, help='Decide or not to resample the images to a new resolution')
parser.add_argument("--new_resolution", type=float, default=(1.5, 1.5, 1.5), help='New resolution')
parser.add_argument("--input_channels", type=float, nargs=1, default=1, help="Input channels")
parser.add_argument("--output_channels", type=float, nargs=1, default=1, help="Output channels (Current implementation supports one output channel")
parser.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 64], help="Input dimension for the generator")
parser.add_argument("--batch_size", type=int, nargs=1, default=1, help="Batch size to feed the network (currently supports 1)")
# Inference parameters
parser.add_argument("--stride_inplane", type=int, nargs=1, default=8, help="Stride size in 2D plane")
parser.add_argument("--stride_layer", type=int, nargs=1, default=8, help="Stride size in z direction")
args = parser.parse_args()

if args.Use_GPU is True:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.Select_GPU)

def prepare_batch(image, ijk_patch_indices):
    image_batches = []
    for batch in ijk_patch_indices:
        image_batch = []
        for patch in batch:
            image_patch = image[patch[0]:patch[1], patch[2]:patch[3], patch[4]:patch[5]]
            image_batch.append(image_patch)

        image_batch = np.asarray(image_batch)
        image_batch = image_batch[:, :, :, :, np.newaxis]
        image_batches.append(image_batch)

    return image_batches

# segment single image
def image_evaluate(model, image_path, result_path, resample, resolution, patch_size_x, patch_size_y, patch_size_z, stride_inplane, stride_layer, batch_size=1):

    # create transformations to image and labels
    transforms = [
        # NiftiDataset.Normalization(),  # not with PET
        # NiftiDataset.StatisticalNormalization(2.5),  # not with PET
        NiftiDataset.Resample(resolution, resample),
        NiftiDataset.Padding((patch_size_x, patch_size_y, patch_size_z))
    ]

    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    # read image file
    reader = sitk.ImageFileReader()
    reader.SetFileName(image_path)
    image = reader.Execute()

    normalizeFilter = sitk.NormalizeImageFilter()
    resacleFilter = sitk.RescaleIntensityImageFilter()
    resacleFilter.SetOutputMaximum(255)
    resacleFilter.SetOutputMinimum(0)
    image = normalizeFilter.Execute(image)  # set mean and std deviation
    image = resacleFilter.Execute(image)  # set intensity 0-65535

    # preprocess the image and label before inference
    image_tfm = image

    # create empty label in pair with transformed image
    label_tfm = sitk.Image(image_tfm.GetSize(), sitk.sitkUInt8)
    label_tfm.SetOrigin(image_tfm.GetOrigin())
    label_tfm.SetDirection(image.GetDirection())
    label_tfm.SetSpacing(image_tfm.GetSpacing())

    sample = {'image': image_tfm, 'label': label_tfm}

    for transform in transforms:
        sample = transform(sample)

    image_tfm, label_tfm = sample['image'], sample['label']

    # convert image to numpy array
    image_np = sitk.GetArrayFromImage(image_tfm)
    label_np = sitk.GetArrayFromImage(label_tfm)

    label_np = np.asarray(label_np, np.float32)

    # unify numpy and sitk orientation
    image_np = np.transpose(image_np, (2, 1, 0))
    label_np = np.transpose(label_np, (2, 1, 0))

    # a weighting matrix will be used for averaging the overlapped region
    weight_np = np.zeros(label_np.shape)

    # prepare image batch indices
    inum = int(math.ceil((image_np.shape[0] - patch_size_x) / float(stride_inplane))) + 1
    jnum = int(math.ceil((image_np.shape[1] - patch_size_y) / float(stride_inplane))) + 1
    knum = int(math.ceil((image_np.shape[2] - patch_size_z) / float(stride_layer))) + 1

    patch_total = 0
    ijk_patch_indices = []
    ijk_patch_indicies_tmp = []

    for i in range(inum):
        for j in range(jnum):
            for k in range(knum):
                if patch_total % batch_size == 0:
                    ijk_patch_indicies_tmp = []

                istart = i * stride_inplane
                if istart + patch_size_x > image_np.shape[0]:  # for last patch
                    istart = image_np.shape[0] - patch_size_x
                iend = istart + patch_size_x

                jstart = j * stride_inplane
                if jstart + patch_size_y > image_np.shape[1]:  # for last patch
                    jstart = image_np.shape[1] - patch_size_y
                jend = jstart + patch_size_y

                kstart = k * stride_layer
                if kstart + patch_size_z > image_np.shape[2]:  # for last patch
                    kstart = image_np.shape[2] - patch_size_z
                kend = kstart + patch_size_z

                ijk_patch_indicies_tmp.append([istart, iend, jstart, jend, kstart, kend])

                if patch_total % batch_size == 0:
                    ijk_patch_indices.append(ijk_patch_indicies_tmp)

                patch_total += 1

    batches = prepare_batch(image_np, ijk_patch_indices)

    # acutal segmentation
    for i in tqdm(range(len(batches))):
        batch = batches[i]

        pred = model.predict(batch, verbose=2, batch_size=1)  # predict segmentation
        pred = np.squeeze(pred, axis=4)

        istart = ijk_patch_indices[i][0][0]
        iend = ijk_patch_indices[i][0][1]
        jstart = ijk_patch_indices[i][0][2]
        jend = ijk_patch_indices[i][0][3]
        kstart = ijk_patch_indices[i][0][4]
        kend = ijk_patch_indices[i][0][5]
        label_np[istart:iend, jstart:jend, kstart:kend] += pred[0, :, :, :]
        weight_np[istart:iend, jstart:jend, kstart:kend] += 1.0

    print("{}: Evaluation complete".format(datetime.datetime.now()))
    # eliminate overlapping region using the weighted value
    label_np = np.rint(np.float32(label_np) / np.float32(weight_np) + 0.01)

    # convert back to sitk space
    label_np = np.transpose(label_np, (2, 1, 0))

    # convert label numpy back to sitk image
    label_tfm = sitk.GetImageFromArray(label_np)
    label_tfm.SetOrigin(image_tfm.GetOrigin())
    label_tfm.SetDirection(image.GetDirection())
    label_tfm.SetSpacing(image_tfm.GetSpacing())

    # save segmented label
    writer = sitk.ImageFileWriter()

    if resample is True:

        label = resample_sitk_image(label_tfm, spacing=image.GetSpacing(), interpolator='linear')
        label = resize(label, (sitk.GetArrayFromImage(image)).shape, sitk.sitkLinear)
        label_np = sitk.GetArrayFromImage(label)
        label.SetDirection(image.GetDirection())
        label.SetOrigin(image.GetOrigin())
        label.SetSpacing(image.GetSpacing())

    else:
        label = label_tfm

    print("{}: Resampling label back to original image space...".format(datetime.datetime.now()))
    label_directory = os.path.join(result_path)
    writer.SetFileName(label_directory)
    writer.Execute(label)
    print("{}: Save evaluate label at {} success".format(datetime.datetime.now(), result_path))


input_dim = [args.batch_size,  args.patch_size[0],  args.patch_size[1], args.patch_size[2], args.input_channels]
model = UNetGenerator(input_dim=input_dim)

image_evaluate(model, args.image, args.result, args.resample, args.new_resolution, args.patch_size[0],args.patch_size[1],args.patch_size[2], args.stride_inplane, args.stride_layer)
