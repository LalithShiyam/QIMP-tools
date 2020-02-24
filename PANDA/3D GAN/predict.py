#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from utils.NiftiDataset import *
import utils.NiftiDataset as NiftiDataset
from tqdm import tqdm
import datetime
import math


def from_numpy_to_itk(image_np,image_itk):
    image_np = np.transpose(image_np, (2, 1, 0))
    image = sitk.GetImageFromArray(image_np)
    image.SetOrigin(image_itk.GetOrigin())
    image.SetDirection(image_itk.GetDirection())
    image.SetSpacing(image_itk.GetSpacing())
    return image


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
def segment_image_evaluate(model, image_path, resample, resolution,  crop_background, patch_size_x, patch_size_y, patch_size_z, stride_inplane, stride_layer, batch_size):

    # -------------- create transformations to image and labels -----------------------
    transforms = [
        NiftiDataset.Resample(resolution, resample),
        NiftiDataset.Padding((patch_size_x, patch_size_y, patch_size_z))
    ]

    case = image_path
    case = case.split('/')
    case = case[3]
    case = case.split('.')
    case = case[0]

    # case = image_path
    # case = case.split('/')
    # case = case[2]
    # case = case.split('.')
    # case = case[0]
    # case = case.split('\\')
    # case = case[1]

    if not os.path.isdir('./Data_folder/results'):
        os.mkdir('./Data_folder/results')

    label_directory = os.path.join('./Data_folder/results', case)

    if not os.path.isdir(label_directory):  # create folder
        os.mkdir(label_directory)

    # normalize the image
    normalizeFilter = sitk.NormalizeImageFilter()
    resacleFilter = sitk.RescaleIntensityImageFilter()
    resacleFilter.SetOutputMaximum(255)
    resacleFilter.SetOutputMinimum(0)

    # read image file
    reader = sitk.ImageFileReader()
    reader.SetFileName(image_path)
    image = reader.Execute()
    image = normalizeFilter.Execute(image)  # set mean and std deviation
    image = resacleFilter.Execute(image)


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

    # ----------------- Padding the image if the z dimension is not even ----------------------
    image_np = sitk.GetArrayFromImage(image_tfm)
    image_np = np.transpose(image_np, (2, 1, 0))

    if (image_np.shape[2] % 2) == 0:
        image_tfm = image_tfm
        label_tfm = label_tfm
        Padding = False
    else:
        image_np = np.pad(image_np, ((0, 0), (0, 0), (0, 1)), 'constant')
        image_tfm = from_numpy_to_itk(image_np, image_tfm)

        # create empty label in pair with transformed image
        label_tfm = sitk.Image(image_tfm.GetSize(), sitk.sitkUInt8)
        label_tfm.SetOrigin(image_tfm.GetOrigin())
        label_tfm.SetDirection(image_tfm.GetDirection())
        label_tfm.SetSpacing(image_tfm.GetSpacing())
        Padding = True

    # ----------------- Computing centroid of the image to crop the background -------------------
    threshold = sitk.BinaryThresholdImageFilter()
    threshold.SetLowerThreshold(1)
    threshold.SetUpperThreshold(255)
    threshold.SetInsideValue(1)
    threshold.SetOutsideValue(0)

    roiFilter = sitk.RegionOfInterestImageFilter()
    roiFilter.SetSize([crop_background[0], crop_background[1], crop_background[2]])

    if patch_size_x > crop_background[0]:
        print('patch size x bigger than image dimension x')
        quit()
    if patch_size_y > crop_background[1]:
        print('patch size y bigger than image dimension y')
        quit()
    if patch_size_z > crop_background[2]:
        print('patch size y bigger than image dimension y')
        quit()

    image_mask = threshold.Execute(image_tfm)
    image_mask = sitk.GetArrayFromImage(image_mask)
    image_mask = np.transpose(image_mask, (2, 1, 0))

    # centroid of the brain input for the inference
    centroid = scipy.ndimage.measurements.center_of_mass(image_mask)

    x_centroid = np.int(centroid[0])
    y_centroid = np.int(centroid[1])

    roiFilter.SetIndex([int(x_centroid - (crop_background[0]) / 2), int(y_centroid - (crop_background[1]) / 2), 0])
    start_x_cropping = (int(x_centroid - (crop_background[0]) / 2))
    start_y_cropping = (int(y_centroid - (crop_background[1]) / 2))
    # ------------------------------------------------------------------------------------------------

    # cropping the background
    image_tfm = roiFilter.Execute(image_tfm)
    label_tfm = roiFilter.Execute(label_tfm)

    # convert image to numpy array
    image_np = sitk.GetArrayFromImage(image_tfm).astype(np.uint8)
    label_np = sitk.GetArrayFromImage(label_tfm).astype(np.uint8)

    label_np = np.asarray(label_np, np.float32)

    # unify numpy and sitk orientation
    image_np = np.transpose(image_np, (2, 1, 0))
    label_np = np.transpose(label_np, (2, 1, 0))

    # a weighting matrix will be used for averaging the overlapped region
    weight_np = np.zeros(label_np.shape)

    # ---------------------- Prepare image batch indices---------------------------------
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

    # ------------------------ Inference  GAN ---------------------------------------
    for i in tqdm(range(len(batches))):
        batch = batches[i]

        pred = model.predict(batch, verbose=2, batch_size=1)  # prediction GAN
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

    # ------------------- Coming back to (344,344,127) dimension ----------------------------------------
    if Padding is True:
        label_np = label_np[:, :, 0:(label_np.shape[2]-1)]

    label = sitk.GetArrayFromImage(image)
    label = np.transpose(label, (2, 1, 0))

    label[start_x_cropping:start_x_cropping + crop_background[0], start_y_cropping:start_y_cropping + crop_background[1], :] = label_np
    label = from_numpy_to_itk(label, image)
    # --------------------------------------------------------------------------------------------------

    # save segmented label
    writer = sitk.ImageFileWriter()

    if resample is True:

        label = resample_sitk_image(label, spacing=image.GetSpacing(), interpolator='linear')
        label = resize(label, (sitk.GetArrayFromImage(image)).shape[::-1], sitk.sitkLinear)
        label.SetDirection(image.GetDirection())
        label.SetOrigin(image.GetOrigin())
        label.SetSpacing(image.GetSpacing())

    else:
        label = label

    print("{}: Resampling label back to original image space...".format(datetime.datetime.now()))
    label_directory = os.path.join(label_directory, 'label_prediction.nii.gz')
    writer.SetFileName(label_directory)
    writer.Execute(label)
    # print("{}: Save evaluate label at {} success".format(datetime.datetime.now(), label_path))
    print('************* Next image coming... *************')


def check_accuracy_model(model, images_list, resample, new_resolution, crop_background, patch_size_x, patch_size_y, patch_size_z, stride_inplane, stride_layer, batch_size):

    model = model

    f = open(images_list, 'r')
    images = f.readlines()
    f.close()


    print("0/%i (0%%)" % len(images))
    for i in range(len(images)):

       segment_image_evaluate(model=model, image_path=images[i].rstrip(), resample= resample, resolution=new_resolution, crop_background=crop_background, patch_size_x=patch_size_x,
                                        patch_size_y=patch_size_y, patch_size_z=patch_size_z,  stride_inplane=stride_inplane, stride_layer=stride_layer, batch_size=batch_size)






