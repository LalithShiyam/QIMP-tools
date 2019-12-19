![Panda-logo](Images/Panda-logo.png)

# PANDA: PET nAvigators usiNg Deep leArning

PANDA pipeline, is a computational toolbox (MATLAB + python) for generating PET navigators using Generative Adversarial networks. 

# Workflow

![PANDA-workflow](Images/PANDA-workflow.png)

# Examples

![Low dose](C:\Users\diommi56\Documents\GitHub\QIMP-tools\PANDA\Images\gan_axial.gif.gif) ![Low dose](Images/gan_axial.gif)

# Requirements

- MATLAB R2016a or higher
- SPM 12
- Python 3
- github repo: git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

# MATLAB scripts and their function 

- checkFileFormat.m : Checks the format of the medical images in a given folder, the output is a string which says if the folder contains 'Dicom', 'Nifti' or 'Analyze'.

- convertDicomtoNii.m: Converts the Dicom series in a folder to nifti files, using SPM.

- cropNiftiForGan.m: Crops a PET nifti file of 344 x 344 x 127 into GAN compatible matrix size 256 x 256 x 128 (U-net dependency)

- prepMedImgForGan.m: This script was mainly created for converting 3D PET images into 2D png or jpg images for using open-source tools of non-medical images.

- prepDataForTraining.m: A lazy function which i wrote for sorting the converted 2D png or jpg images into 'test','train', and 'val' (validation) folders. The ratio is automatically defined: 60% of total datasets for training, 20% for testing and 20% validation.

- convertGanOutputToNativeSpace.m: Converts GAN Nifti files to native Nifti files which are compatible for image registration (344 x 344 x 127).

- callPytorchFor2DGAN.m: Creates an '.command' file to run the pytorch scripts for generating image-pairs (A-B)

- createGIF.m: creates a GIF animation from a series of time-series images (ideally).

- removeEmptySlices.m: Removes empty image pairs before GAN training (prevents unwanted calculation).


# Python scripts and their function

- Niftitest.py: Runs the 2D based Pix2pix inference on the .nii volumes and returns the late dose frame from the low dose frames. 

- Normalize_PET_images: Normalizes and crops the images for the training. To run before the main.py

- data_generator.py / NiftiDataset.py : They augment the data, extract the patches and feed them to the GAN. 

- check_loader_patches: Shows the low dose and high dose patches fed to the Gan during the training  

- generator.py / discriminator.py / DCGAN.py: the architecture of the GAN.

- main.py: Runs the training and the prediction on the training and validation dataset.

- logger.py: Generates sample images and histograms to monitor the training.

- predict.py: It launches the inference on training and validation data in the main.py

- predict_single_image.py: It launches the inference on a single input image chosen by the user.

