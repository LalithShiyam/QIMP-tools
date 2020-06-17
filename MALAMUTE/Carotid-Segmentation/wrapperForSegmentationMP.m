%% Purpose
% This is a wrapper script for running the 'coreSegmentationMP.m' program, 
% which is responsible for automatic petrous segmentation.
%%
%% Author information
% Lalith Kumar Shiyam Sundar, 
% Quantitative Imaging and Medical Physics, Medical University of Vienna

%% Adapted 06/2020
% core program for extracting a simple petrous mask out of a contrast
% enhanced T1-brain-MR by Matthias

%% Inputs to be entered by the user                  
% coreSegInputs.pathOfMR - Physical path of the conrast enhanced
% t1-brain-MR in DICOM-format
% coreSegInputs.patientCode - Recognition Name trough out the programm,
% choose as you want
% coreSegInputs.path2storeSeg - Physical path where the resulting petrous
% mask will be stored

%% Program start
% Copy your physical path of the 3D time-of-flight MR angiography DICOM
% images.

coreSegInputs.pathOfMR = 'C:\Users\Lalith shiyam\Desktop\Matthias\FET_TEST\T1_HEAD_T1_MPRAGE_TRA_ISO_EGN_KM_0037'; % To be filled in by the user!
coreSegInputs.patientCode = 'P001';
coreSegInputs.path2StoreSeg= 'C:\Users\Lalith shiyam\Desktop\Matthias\FET_TEST\SegmentationOutput';
%% Hard-coded variables.

% Error messages

errorMsg{1}='This program needs a path to the "DICOM" series!';
errorMsg{2}='Raise an issue in github...';


%% Preliminary checks are being done here, to see if the program can be run.
fileFormat=checkFileFormat(coreSegInputs.pathOfAngio); % check if the folder has dicom images.
switch fileFormat 
    case 'Dicom'
        disp(['DICOM images found in ',coreSegInputs.pathOfAngio,'...']);
        disp('Applying segmentation algorithm on the dataset...');
       coreSegmentationMP(coreSegInputs); % Running the coreSegmentation algorithm
    otherwise
        error(errorMsg{1});
        error(errorMsg{2});
end

%%


