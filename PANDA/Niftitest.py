""" Pix2pix gan Inference on .Nii files

The script run the trained Pix2pix model on the .nii files in the folder ./nifti
The inference files will be overwritten with the same name in the same folder.
The pix2pix model is uploaded and the inference is run on each slice on the input volume; the slices are then
packed together and the .nii file is created.

The script needs:
  - location of Niftitest.py in the same folder of train.py and test.py files
  - the weights of the pix2pix model in the path: "checkpoints/experiment_name/"
  - create folder './nifti' with the .nii with the nifti files to run the inference
  - To modify the "options/base_options.py" file:
               '--dataroot', required=False, default='./nifti'
               '--model', type=str, default='pix2pix'
               '--netG', type=str, default='unet_256'
               '--dataset_mode', type=str, default='single'

Usage: python Niftytest.py

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
parser.add_argument("--low_dose_root", default='./nifti', help='path to the .nii files folder to run the inference')
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


opt = TestOptions().parse()  # get test options
opt.num_threads = 0   # test code only supports num_threads = 1
opt.batch_size = 1    # test code only supports batch_size = 1
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers
if opt.eval:
    model.eval()


config = dict()
# config["images_folder"] = './datasets/volumes/'  # Folder where there are the data
config["images_folder"] = args.low_dose_root  # Folder where there are the data
images = lstFiles(config["images_folder"])


def inference(image_name):

    image = sitk.ReadImage(image_name)           # read the image
    array = sitk.GetArrayFromImage(image)        # get array of the image
    array = np.transpose(array, axes=(2, 1, 0))  # reshape array from itk z,y,x  to x,y,z

    high_dose_array = np.zeros((array.shape[0], array.shape[1], 1))

    for i in range(array.shape[2]):
        slice = array[:,:,i]
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



