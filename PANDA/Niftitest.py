""" Pix2pix gan Inference on .Nii files

The script run the trained Pix2pix model on the .nii files in the folder selected by the user
The inference files will be overwritten with the same name in the same folder.
The pix2pix model is uploaded and the inference is run on each slice on the input volume; the slices are then
packed together and the .nii file is created.

Usage: python Niftytest.py --dataroot [name] --model pix2pix --netG [name] --dataset_mode single --checkpoints_dir [name]

                           --dataroot: path where the .nii files are stored
       !!!(not working)    (--model: pix2pix (not working))
       pix2pix needs to    --netG: unet_256 (if the image 256x256)
       be default          --dataset_mode: single
                           --checkpoints_dir: folder path where the --name folder is
                           --name: folder name where the weights are stored for the inference' (needs to be a subfolder of checkpoints)


Example: python Niftitest.py --dataroot ./nifti --netG unet_256 --dataset_mode single --checkpoints_dir ./checkpoints --name experiment_name

"""

import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataroot", default='./nifti', help='path to the .nii files folder to run the inference')
# parser.add_argument("--model", default='pix2pix', help='model to run')
parser.add_argument("--netG", default='unet_256', help='specify generator architecture')
parser.add_argument("--dataset_mode", default='single', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
parser.add_argument("--checkpoints_dir", default='./checkpoints', help='model weights are saved here')
parser.add_argument("--name", default='experiment_name', help='model weights are saved here')
args = parser.parse_args()


def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts
def lstFiles(Path):

    images_list = []  # create an empty list, the raw image data files is stored here
    for dirName, subdirList, fileList in os.walk(Path):
        for filename in fileList:
            if ".nii" in filename.lower():
                images_list.append(os.path.join(dirName, filename))

    images_list = sorted(images_list, key=numericalSort)
    return images_list


config = dict()
config["images_folder"] = args.dataroot  # Folder where there are the data
images = lstFiles(config["images_folder"])

opt = TestOptions().parse()  # get test options
opt.dataroot = args.dataroot
# opt.model = args.model
opt.netG = args.netG
opt.dataset_mode = args.dataset_mode
opt.checkpoints_dir = args.checkpoints_dir
opt.name = args.name
opt.num_threads = 0   # test code only supports num_threads = 1
opt.batch_size = 1    # test code only supports batch_size = 1
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers
if opt.eval:
    model.eval()


def inference(image_name):

    image = sitk.ReadImage(image_name)           # read the image
    array = sitk.GetArrayFromImage(image)        # get array of the image
    array = np.transpose(array, axes=(2, 1, 0))  # reshape array from itk z,y,x  to x,y,z

    high_dose_array = np.zeros((array.shape[0], array.shape[1], 1))

    for i in range(array.shape[2]):
        slice = array[:, :, i]
        matplotlib.pyplot.imsave((os.path.join(config["images_folder"], "inference.png")), slice)
        dataset = create_dataset(opt)
        for i, data in enumerate(dataset):
            if i >= opt.num_test:  # only apply our model to opt.num_test images.
                break
            model.set_input(data)                                            # unpack data from data loader
            model.test()                                                     # run inference
            visuals = model.get_current_visuals()                            # get image results
            result = visuals['fake']
            result = result.cpu().numpy()
            result=result[0]
            result=result[0]
            result= result[:,:,np.newaxis]
            high_dose_array=np.append(high_dose_array, result, axis=2)      # append inference for each slice to build the volume

    high_dose_array = high_dose_array[1:]                                   # create the .nii file from the inference
    high_dose_array = np.transpose(high_dose_array, axes=(2, 1, 0))         # reshape array to z,y,x
    high_dose = sitk.GetImageFromArray(high_dose_array)
    high_dose.SetOrigin(image.GetOrigin())
    high_dose.SetDirection(image.GetDirection())
    high_dose.SetSpacing(image.GetSpacing())

    sitk.WriteImage(high_dose, image_name)


if __name__ == '__main__':

    for i in range(len(images)):
        inference(images[i])

    print('Inference completed for all images')
    os.remove(os.path.join(config["images_folder"], "inference.png"))



