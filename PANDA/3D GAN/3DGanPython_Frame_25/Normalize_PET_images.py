from utils.NiftiDataset import *
import argparse

''' The script normalize the images in the low dose and high dose folders. Normalization is performed and images are scaled to interval values: 0-255.
    The images will be overwritten to the previous ones
    In case you want to change the scale, please take in consideration to change or not the activation function of the generator.
    The images are cropped from (256,256,128) to (128,128,128)'''


parser = argparse.ArgumentParser()
parser.add_argument("--images_folder", type=str, default='./Data_folder/volumes', help='path to the .nii low dose images')
parser.add_argument("--labels_folder", type=str, default='./Data_folder/labels', help='path to the .nii high dose images')
args = parser.parse_args()


def normalize_Pet_image(image_name):

    print('normalizing', image_name)
    image = sitk.ReadImage(image_name)
    normalizeFilter = sitk.NormalizeImageFilter()
    resacleFilter = sitk.RescaleIntensityImageFilter()
    resacleFilter.SetOutputMaximum(255)
    resacleFilter.SetOutputMinimum(0)
    image = normalizeFilter.Execute(image)  # set mean and std deviation
    image = resacleFilter.Execute(image)

    label_np = sitk.GetArrayFromImage(image).astype(np.uint8)
    label_np = np.transpose(label_np, axes=(2, 1, 0))
    label_np = label_np[64:192, 76:204, 0:128]         # cropping
    label_np = np.transpose(label_np, axes=(2, 1, 0))

    label_tfm = sitk.GetImageFromArray(label_np)
    label_tfm.SetOrigin(image.GetOrigin())
    label_tfm.SetDirection(image.GetDirection())
    label_tfm.SetSpacing(image.GetSpacing())

    sitk.WriteImage(label_tfm, image_name)


images = lstFiles(args.images_folder)
labels = lstFiles(args.labels_folder)

for i in range(len(images)):
    normalize_Pet_image(images[i])

for i in range(len(labels)):
    normalize_Pet_image(labels[i])

