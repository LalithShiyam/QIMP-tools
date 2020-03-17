import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import numpy as np
from utils.NiftiDataset import *
import utils.NiftiDataset as NiftiDataset
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


def plot_generated_batch(image,label, model, resample, resolution, crop_background, patch_size_x, patch_size_y, patch_size_z, stride_inplane, stride_layer, batch_size,epoch):

    # --------------- reading the images from the validation list txt --------------
    f = open(image, 'r')
    images = f.readlines()
    f.close()

    f = open(label, 'r')
    labels = f.readlines()
    f.close()

    image = images[2].rstrip()  # taking the first in the list to monitor during the training
    label = labels[2].rstrip()
    # ------------------------------------------------------------------------------


    transforms = [
        NiftiDataset.Resample(resolution, resample),
        NiftiDataset.Padding((patch_size_x, patch_size_y, patch_size_z))
    ]

    # normalize the images
    normalizeFilter = sitk.NormalizeImageFilter()
    resacleFilter = sitk.RescaleIntensityImageFilter()
    resacleFilter.SetOutputMaximum(255)
    resacleFilter.SetOutputMinimum(0)

    reader = sitk.ImageFileReader()
    reader.SetFileName(image)
    image = reader.Execute()
    image = normalizeFilter.Execute(image)  # set low dose mean and std deviation
    image = resacleFilter.Execute(image)

    reader = sitk.ImageFileReader()
    reader.SetFileName(label)
    label = reader.Execute()
    label = normalizeFilter.Execute(label)  # set high dose mean and std deviation
    label = resacleFilter.Execute(label)

    # create empty label in pair with transformed image
    image_tfm = image
    label_tfm = sitk.Image(image_tfm.GetSize(), sitk.sitkFloat32)
    label_tfm.SetOrigin(image_tfm.GetOrigin())
    label_tfm.SetDirection(image.GetDirection())
    label_tfm.SetSpacing(image_tfm.GetSpacing())

    original = {'image': image_tfm, 'label': label}
    sample = {'image': image_tfm, 'label': label_tfm}

    for transform in transforms:
        sample = transform(sample)

    image_tfm, label_tfm = sample['image'], sample['label']
    label_true = original['label']

    # ----------------------  Padding the image if the z dimension is not even ----------------------
    image_np = sitk.GetArrayFromImage(image_tfm)
    image_np = np.transpose(image_np, (2, 1, 0))

    if (image_np.shape[2] % 2) == 0:
        image_tfm = image_tfm
        label_tfm = label_tfm
        label_true = label_true
    else:
        # pad low dose image
        image_np = np.pad(image_np, ((0, 0), (0, 0), (0, 1)), 'constant')
        image_tfm = from_numpy_to_itk(image_np, image_tfm)

        # pad high dose image
        label_true_np = sitk.GetArrayFromImage(label_true)
        label_true_np = np.transpose(label_true_np, (2, 1, 0))
        label_true_np = np.pad(label_true_np, ((0, 0), (0, 0), (0, 1)), 'constant')
        label_true = from_numpy_to_itk(label_true_np, image_tfm)

        # create empty label in pair with transformed image
        label_tfm = sitk.Image(image_tfm.GetSize(), sitk.sitkFloat32)
        label_tfm.SetOrigin(image_tfm.GetOrigin())
        label_tfm.SetDirection(image_tfm.GetDirection())
        label_tfm.SetSpacing(image_tfm.GetSpacing())

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
        print('patch size y bigger than image dimension z')
        quit()

    image_mask = threshold.Execute(label_true)
    image_mask = sitk.GetArrayFromImage(image_mask)
    image_mask = np.transpose(image_mask, (2, 1, 0))

    # centroid of the brain input for the inference
    centroid = scipy.ndimage.measurements.center_of_mass(image_mask)
    x_centroid = np.int(centroid[0])
    y_centroid = np.int(centroid[1])
    roiFilter.SetIndex([int(x_centroid - (crop_background[0]) / 2), int(y_centroid - (crop_background[1]) / 2), 0])
    # ------------------------------------------------------------------------------------------------

    # cropping the background from low dose, high dose and gan image
    image_tfm = roiFilter.Execute(image_tfm)
    label_tfm = roiFilter.Execute(label_tfm)
    label_true = roiFilter.Execute(label_true)

    # ****************************
    low = sitk.GetArrayFromImage(image_tfm)    # values to plot on the histograms
    high = sitk.GetArrayFromImage(label_true)
    # ****************************

    # convert image to numpy array
    image_np = sitk.GetArrayFromImage(image_tfm)
    label_np = sitk.GetArrayFromImage(label_tfm)
    label_np = np.asarray(label_np, np.float32)

    slice_volume_20 = image_np[20]
    slice_volume_40 = image_np[40]
    slice_volume_63 = image_np[63]
    slice_volume_80 = image_np[80]

    x = sitk.GetArrayFromImage(label_true)

    slice_label_20 = x[20]
    slice_label_40 = x[40]
    slice_label_63 = x[63]
    slice_label_80 = x[80]

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
    for i in range(len(batches)):
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
    label_np = (np.float32(label_np) / np.float32(weight_np) + 0.01)

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

        label = resample_sitk_image(label_tfm, spacing=image.GetSpacing(), interpolator='bspline')
        # label = resize(label, (sitk.GetArrayFromImage(image)).shape[::-1], sitk.sitkBSpline)
        label_np = sitk.GetArrayFromImage(label)
        label.SetDirection(image.GetDirection())
        label.SetOrigin(image.GetOrigin())
        label.SetSpacing(image.GetSpacing())

    else:
        label = label_tfm

    label_directory = 'History/Epochs_training/epoch_%s' % epoch
    if not os.path.exists(label_directory):
        os.makedirs(label_directory)
    label_directory = os.path.join(label_directory, 'epoch_prediction.nii.gz')
    writer.SetFileName(label_directory)
    writer.Execute(label)

    # --------------------------- Plotting samples during training ---------------------------------

    slice_predicted_20 = sitk.GetArrayFromImage(label)[20]
    slice_predicted_40 = sitk.GetArrayFromImage(label)[40]
    slice_predicted_63 = sitk.GetArrayFromImage(label)[63]
    slice_predicted_80 = sitk.GetArrayFromImage(label)[80]

    fig = plt.figure()
    fig.set_size_inches(12, 12)

    plt.subplot(5, 3, 1), plt.imshow(slice_volume_20, 'gray'), plt.axis('off'), plt.title('Low dose')
    plt.subplot(5, 3, 2), plt.imshow(slice_predicted_20, 'gray'), plt.axis('off'), plt.title('GAN')
    plt.subplot(5, 3, 3), plt.imshow(slice_label_20, 'gray'), plt.axis('off'), plt.title('High dose')

    plt.subplot(5, 3, 4), plt.imshow(slice_volume_40, 'gray'), plt.axis('off'), plt.title('Low dose')
    plt.subplot(5, 3, 5), plt.imshow(slice_predicted_40, 'gray'), plt.axis('off'), plt.title('GAN')
    plt.subplot(5, 3, 6), plt.imshow(slice_label_40, 'gray'), plt.axis('off'), plt.title('High dose')

    plt.subplot(5, 3, 7), plt.imshow(slice_volume_63, 'gray'), plt.axis('off'), plt.title('Low dose')
    plt.subplot(5, 3, 8), plt.imshow(slice_predicted_63, 'gray'), plt.axis('off'), plt.title('GAN')
    plt.subplot(5, 3, 9), plt.imshow(slice_label_63, 'gray'), plt.axis('off'), plt.title('High dose')

    plt.subplot(5, 3, 10), plt.imshow(slice_volume_80, 'gray'), plt.axis('off'), plt.title('Low dose')
    plt.subplot(5, 3, 11), plt.imshow(slice_predicted_80, 'gray'), plt.axis('off'), plt.title('GAN')
    plt.subplot(5, 3, 12), plt.imshow(slice_label_80, 'gray'), plt.axis('off'), plt.title('High dose')

    plt.subplot(5, 3, 13, autoscale_on=True), plt.hist(low.flatten(), bins=256, range=(3, (low.flatten()).max()), density=0, facecolor='red', align='right', alpha=0.75,
                                                       histtype='stepfilled'), plt.title('Low dose histogram')
    plt.subplot(5, 3, 14, autoscale_on=True), plt.hist(label_np.flatten(), bins=256, range=(3, (label_np.flatten()).max()), density=0, facecolor='red', align='right', alpha=0.75,
                                                       histtype='stepfilled'), plt.title('GAN histogram')
    plt.subplot(5, 3, 15, autoscale_on=True), plt.hist(high.flatten(), bins=256, range=(3, (high.flatten()).max()), density=0, facecolor='red', align='right', alpha=0.75,
                                                       histtype='stepfilled'), plt.title('High dose histogram')

    plt.savefig('History/Epochs_training/epoch_%s.png' % epoch)
    plt.close()


