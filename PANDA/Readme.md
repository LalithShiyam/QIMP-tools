![Panda-logo](Images/Panda-logo.png)

# PANDA: PET nAvigators usiNg Deep leArning

PANDA pipeline, is a computational toolbox (MATLAB + python) for generating PET navigators using Generative Adversarial networks.

# Requirements

- MATLAB R2016a or higher
- SPM 12

# MATLAB scripts and their function

- checkFileFormat.m : Checks the format of the medical images in a given folder, the output is a string which says if the folder contains 'Dicom', 'Nifti' or 'Analyze'.

- convertDicomtoNii.m: Converts the Dicom series in a folder to nifti files, using SPM.

- cropNiftiForGan.m: Crops a PET nifti file of 344 x 344 x 127 into GAN compatible matrix size 256 x 256 x 256 (U-net dependency)

- prepMedImgForGan.m: This script was mainly created for converting 3D PET images into 2D png or jpg images for using open-source tools of non-medical images.

- prepDataForTraining.m: A lazy function which i wrote for sorting the converted 2D png or jpg images into 'test','train', and 'val' (validation) folders. The ratio is automatically defined: 60% of total datasets for training, 20% for testing and 20% validation.

- convertGanOutputToNativeSpace.m: Converts GAN Nifti files to native Nifti files which are compatible for image registration (344 x 344 x 127).

- callPytorchFor2DGAN.m: Creates an '.command' file to run the pytorch scripts for generating image-pairs (A-B)

- Niftitest.py: Run the Pix2pix inference on the .nii volumes and returns the late dose frame from the low dose frames 
