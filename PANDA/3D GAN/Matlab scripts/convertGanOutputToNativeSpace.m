% This function converts GAN Nifti files to native Nifti files which
% are compatible for image registration (344 x 344 x 127)
%
% Author : Lalith Kumar Shiyam Sundar
% Date   : 18 November, 2019
%
% Inputs: 
%       [1]CGONinputs.path2GanNifti: file path to the nifti medical images.
%       [2]CGONinputs.where2Store: file path to store the generated images.
%       [3]CGONinputs.path2OrgNifti: file path of the original nifti images
%
% Outputs: Folder containing the converted images. 
%
% Usage: convertGanOutputToNativeSpace(CGONinputs);
%       
%------------------------------------------------------------------------%
%                               Program start
%------------------------------------------------------------------------%
function []= convertGanOutputToNativeSpace(CGONinputs)

% Hard-coded variables: to be changed according to needs.

    xMax=344; % x dimension
    yMax=344; % y dimension
    zMax=127; % z dimension
    padRange=108; % padding


% create the folder where the cropped images will be stored.

cd(CGONinputs.where2Store)
splitFiles=regexp(CGONinputs.path2GanNifti,filesep,'split');
convertedFolder=[splitFiles{end},'-','native'];
mkdir(convertedFolder); % creating the converted folder for storing
path2ConvFolder=[CGONinputs.where2Store,filesep,convertedFolder];


% load the nifti files.

cd(CGONinputs.path2GanNifti);
niftiFiles=dir('*.nii');
for lp=1:length(niftiFiles)
    imgVol=niftiread(niftiFiles(lp).name);
    newVol{lp}=imgVol(:,:,1:zMax);
end

cd(CGONinputs.path2OrgNifti)
orgNiftiFiles=dir('*.nii');
for lp=1:length(orgNiftiFiles)
    cd(CGONinputs.path2OrgNifti)
    NativeFileName=['PET-Nav-',orgNiftiFiles(lp).name]; 
    nativeVolumes=niftiread(orgNiftiFiles(lp).name);
    nativeVolumes(padRange+1:xMax-padRange,padRange+1:yMax-padRange,:)=newVol{lp};
    hdrInfo=niftiinfo(orgNiftiFiles(lp).name);
    hdrInfo.Description='GAN derived PET navigators';
    niftiwrite(nativeVolumes,NativeFileName,hdrInfo);
    disp(['Writing ',NativeFileName,'...']);
    movefile(NativeFileName,path2ConvFolder);
end


end
