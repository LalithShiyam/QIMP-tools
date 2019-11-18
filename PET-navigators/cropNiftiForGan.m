% % This function converts images in medical imaging formats (Nifti or
% Analyze) to commercially usable formats (JPG, PNG)
% 
% Note: The code was developed internally to nifti images to GAN
% compatible sizes (256 x 256 x 128). Therefore, the code expects at
% somepoint the inherent matrix size of the pet system (344 x 344 x 127).
% Therefore, it is a limitation. But this can be easily overcome with some
% modification.
%
% Author : Lalith Kumar Shiyam Sundar
% Date   : 18 November, 2019
%
% Inputs: 
%       [1]CNGinputs.path2Nifti: file path to the nifti medical images.
%       [2]CNGinputs.where2Store: file path to store the generated images.
%
% Outputs: Folder containing the converted images. 
%
% Usage: cropNiftiForGan(CNGinputs);
%       
%------------------------------------------------------------------------%
%                           Program start
%------------------------------------------------------------------------%
function [] = cropNiftiForGan(CNGinputs)

% create the folder where the cropped images will be stored.

cd(CNGinputs.where2Store)
splitFiles=regexp(CNGinputs.where2Store,filesep,'split');
convertedFolder=[splitFiles{end},'-','cropped'];
path2ConvFolder=[CNGinputs.where2Store,filesep,convertedFolder];

% load the nifti files.

cd(CNGinputs.path2Nifti)
niftiFiles=dir('*.nii');
cropRange=44;
for lp=1:length(niftiFiles)
    hdrInfo=niftiinfo(niftiFiles(lp).name);
    hdrInfo.Description = 'cropped for GANS: Internal use only!';
    croppedFileName=['crpd-',niftiFiles(lp).name];
    hdrInfo.Filename=croppedFileName;
    hdrInfo.ImageSize=[256 256 128];
    imgVol=niftiread(niftiFiles(lp).name);
    croppedVol=imgVol;
    xMax=size(imgVol,1);
    yMax=size(imgVol,2);
    zMax=size(imgVol,3);
    croppedVol=croppedVol(cropRange+1:xMax-cropRange,cropRange+1:yMax-cropRange,:);
    emptySlice=zeros(size(croppedVol(:,:,1)));
    croppedVol(:,:,zMax+1)=emptySlice;
    niftiwrite(croppedVol,croppedFileName,hdrInfo);
    disp(['Writing ',croppedFileName,'...']);
    movefile(croppedFileName,path2ConvFolder);
end


end