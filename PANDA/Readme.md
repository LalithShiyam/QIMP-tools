![Panda-logo](Images/Panda-logo.png)

# PANDA: PET nAvigators usiNg Deep leArning

PANDA pipeline, is a computational toolbox (MATLAB + python) for generating PET navigators using Generative Adversarial networks. 

# Workflow

![PANDA-workflow](Images/PANDA-workflow.png)

# Examples

Sample images (axial and coronal views): on the left side are the early PET frames, in the middle the output of the 3D GAN and on the right side the corresponding late PET frame

![Low dose](Images/low_dose_axial.gif)![GAN](Images/gan_axial.gif)![High dose](Images/high_dose_axial.gif)
*******************************************************************************
![Low dose](Images/low_dose_coronal.gif)![GAN](Images/gan_coronal.gif)![High dose](Images/high_dose_coronal.gif)
*******************************************************************************
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

- data_generator.py / NiftiDataset.py : They normalize, augment the data, extract the patches and feed them to the 3DGAN. 

- check_loader_patches: Shows paired low and high dose patches fed to the 3DGANGan during the training  

- generator.py / discriminator.py / DCGAN.py: the architecture of the 3DGAN.

- main.py: Runs the training and the inference on the training and validation dataset.

- logger.py: Generates sample images and histograms to monitor the training (called by the main.py).

- predict.py: It launches the inference on training and validation data (called by the main.py).

- predict_single_image.py: It launches the inference on a single input image chosen by the user (not called by the main.py).

# Tutorial for 3DGAN

1) Launch the matlab file "convertDicomtoNii.m" to convert Dicom in Nifti format. All images in this study are produced with a PET/MRI-Siemens Biograph mMR. Frames dimensions are 344x344x127. 

2) Place low-dose frames in "Data_folder/volumes" folder and high-dose frames in "Data_folder/labels" folder. Be sure that low/high dose frames are correctly paired in the two folders.

3) Launch the pipeline for training and testing dataset (example): 
```console
python3 main.py --Create_training_test_dataset=True --Do_you_wanna_train=True  --Do_you_wanna_check_accuracy=True --patch_size=(128,128,64)
```
Sample of the logger, which helps to monitor the training process
![logger](epoch_80.PNG)

4) Launch the inference on only one image (example):

```console
python3 predict_single_image.py --image "path to low dose frame" --result "path to high dose to save" --gen_weights "path to the weights of the generator network"  --patch_size=(128,128,64)
```





