![Panda-logo](Images/Panda-logo.JPG)

# PANDA: PET nAvigators usiNg Deep leArning

PANDA (Pytorch) pipeline, is a computational toolbox (MATLAB + pytorch) for generating PET navigators using Generative Adversarial networks. 
This is the Torch version of the first repository of PANDA (https://github.com/LalithShiyam/QIMP-tools/tree/master/PANDA/3D%20GAN) that was developed in Keras

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
- Pytorch
- See requirements.txt list (execute "pip install -r requirements.txt" to install the python dependencies)

# MATLAB scripts and their function 

- checkFileFormat.m : Checks the format of the medical images in a given folder, the output is a string which says if the folder contains 'Dicom', 'Nifti' or 'Analyze'.

- convertDicomtoNii.m: Converts the Dicom series in a folder to nifti files, using SPM.

# Python scripts and their function

- organize_folder_structure.py: Organize the data in the folder structure for the network

- NiftiDataset.py : They augment the data, extract the patches and feed them to the GAN (reads .nii files). NiftiDataset.py
  skeleton taken from https://github.com/jackyko1991/unet3d-pytorch

- check_loader_patches: Shows paired early and late frames patches fed to the 3DGANGan during the training  

- generators.py / discriminators.py : the architectures of the 3DGAN.

- train.py: Runs the training the training dataset.

- logger.py: Generates sample images to monitor the training.

- predict_single_image.py: It launches the inference on a single input image chosen by the user.

# Tutorial for 3DGAN

## Training:

1) Launch the matlab file "convertDicomtoNii.m" to convert Dicom files in Nifti format.

2) Launch organize_folder_structure.py to organize the data for the training. If early-frames  are in in "./volumes" folder and late-frames in "./labels" folder run: 

```console
python organize_folder_structure.py --images ./volumes --labels ./labels --split 2
```

Be sure that early and late frames were correctly paired in the two original "./volumes" and "./labels" folders. The value "split" is the number of samples saved as testing dataset

Resulting folder Structure:

	.
	├── Data_folder                   
	|   ├── train              
	|   |   ├── patient_1             # Training
	|   |   |   ├── image.nii              
	|   |   |   └── label.nii                     
	|   |   └── patient_2             
	|   |   |   ├── image.nii              
	|   |   |   └── label.nii              
	|   ├── test               
	|   |   ├── patient_3             # Testing
	|   |   |   ├── image.nii              
	|   |   |   └── label.nii              
	|   |   └── patient_4             
	|   |   |   ├── image.nii              
	|   |   |   └── label.nii            


3) Launch the pipeline for training and testing dataset (example): 
```console
python3 main.py --Create_training_test_dataset=True --Do_you_wanna_train=True  --Do_you_wanna_check_accuracy=True --patch_size=(128,128,64)
```
Sample of the logger, which helps to monitor the training process
![logger](Images/epoch_80.png)

There are several parameters you can set for the training; you can modify the default ones in the source code or type them from the pipeline. The description for each parameter included is in the main.py source file.
Please open it to read the descriptions if you want to train on your own data.

## Inference:

Launch the inference on only one image (example):

```console
python3 predict_single_image.py --image "path to early frame" --result "path where to save the late frame" --gen_weights "path to the weights of the generator network to load"  --patch_size=(128,128,64)
```
### Sample script inference
```console
C:\Users\David\Desktop\3D GAN>python predict_single_image.py --image C:\Users\David\Desktop\test_image.nii --result C:\Users\David\Desktop\result.nii --gen_weights C:\Users\David\Desktop\weights.h5
```

## Citations

To implement PANDA we were inspired by existing codes from other github contributors. Here is the list of github repositories:

- https://github.com/jackyko1991/vnet-tensorflow

- https://github.com/joellliu/3D-GAN-for-MRI



