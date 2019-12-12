% This function converts native Nifti files to cropped Nifti files which
% are compatible for GANs (128 x 128 x 128)
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

% Hard-coded variables.

xyzDim=[128 128 128]; % dimensions to be cropped
cropRange=108;  % crop region 

% create the folder where the cropped images will be stored.

cd(CNGinputs.where2Store)
splitFiles=regexp(CNGinputs.path2Nifti,filesep,'split');
convertedFolder=[splitFiles{end},'-','cropped'];
mkdir(convertedFolder); % creating the converted folder for storing
path2ConvFolder=[CNGinputs.where2Store,filesep,convertedFolder];

% load the nifti files.

cd(CNGinputs.path2Nifti)
niftiFiles=dir('*.nii');
parfor lp=1:length(niftiFiles)
    hdrInfo=niftiinfo(niftiFiles(lp).name);
    hdrInfo.Description = 'cropped for GANS: Internal use only!';
    croppedFileName=['crpd-',niftiFiles(lp).name];
    hdrInfo.Filename=croppedFileName;
    hdrInfo.ImageSize=xyzDim;
    imgVol=niftiread(niftiFiles(lp).name);
    croppedVol=imgVol;
    xMax=size(imgVol,1);
    yMax=size(imgVol,2);
    zMax=size(imgVol,3);
    croppedVol=croppedVol(cropRange+1:xMax-cropRange,cropRange+1:yMax-cropRange,:);
    emptyVol=zeros([xyzDim(1) xyzDim(2) (xyzDim(3)-size(imgVol,3))]);
    croppedVol=cat(3,croppedVol,int16(emptyVol));
    niftiwrite(croppedVol,croppedFileName,hdrInfo);
    disp(['Writing ',croppedFileName,'...']);
    movefile(croppedFileName,path2ConvFolder);
end


end