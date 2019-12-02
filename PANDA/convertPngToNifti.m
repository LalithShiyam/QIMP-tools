% This function converts GAN generated png files to cropped Nifti files 
% (256 x 256 x 128)
%
% Author : Lalith Kumar Shiyam Sundar
% Date   : 29 November, 2019
%
% Inputs: 
%       [1]CPNinputs.path2CroppedNifti: file path to cropped nifti medical images.
%       [2]CPNinputs.where2Store: file path to store the generated images.
%       [3]CPNinputs.path2GanPngs: file path to GAN generated pngs
%       [4]CPNinputs.subjectID: subjectID to pick (string).
%       [5]CPNinputs.orientation: orientation - axial, sagittal or coronal
%
% Outputs: Folder containing the converted images. 
%
% Usage: convertPngToNifti(CPNinputs);
%       
%------------------------------------------------------------------------%
%                           Program start
%------------------------------------------------------------------------%
function []=convertPngToNifti(CPNinputs)

% Local move variables

path2CroppedNifti=CPNinputs.path2CroppedNifti;
where2Store=CPNinputs.where2Store;
Path2GanPngs=CPNinputs.path2GanPngs;
subjectID=CPNinputs.subjectID;
orientation=CPNinputs.orientation;
fakeVol=zeros([256 256 128]);

% Hard-coded variables

cd(Path2GanPngs);
splitFiles=regexp(CPNinputs.path2GanPngs,filesep,'split');
convertedFolder=[splitFiles{end},'-','nifti-',orientation];
cd(where2Store);
mkdir(convertedFolder); % creating the converted folder for storing
path2ConvFolder=[where2Store,filesep,convertedFolder];

% go to the folder containing GAN png files 

cd(Path2GanPngs)
files2Pick=dir([subjectID,'fake*']);
for lp=1:length(files2Pick)
    fileNames{lp,:}=files2Pick(lp).name;
end
sortedFiles=natsort(fileNames);

% Generate a GAN volume from png files

for lp=1:length(sortedFiles)
    img=imread(sortedFiles{lp,:});
    fakeVol(:,:,lp)=img(:,:,1);
end

% Load the cropped nifti file.

cd(path2CroppedNifti)
subjectID = strrep(subjectID,'_',' ');
niftyFile=dir(subjectID);
ganVolName=['GAN-PET-',niftyFile.name];
croppedNiftiFileInfo=niftiinfo(niftyFile.name);
niftiwrite(int16(fakeVol),ganVolName,croppedNiftiFileInfo);
movefile(ganVolName,path2ConvFolder);
cd(path2ConvFolder)
movefile(ganVolName,niftyFile.name);
end