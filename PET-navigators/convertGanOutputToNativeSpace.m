% This function converts GAN Nifti files to native Nifti files which
% are compatible for image registration (344 x 344 x 127)
%
% Author : Lalith Kumar Shiyam Sundar
% Date   : 18 November, 2019
%
% Inputs: 
%       [1]CGONinputs.path2GanNifti: file path to the nifti medical images.
%       [2]CGONinputs.where2Store: file path to store the generated images.
%
% Outputs: Folder containing the converted images. 
%
% Usage: convertGanOutputToNativeSpace(CGONinputs);
%       
%------------------------------------------------------------------------%
%                               Program start
%------------------------------------------------------------------------%
function []= convertGanOutputToNativeSpace(CGONinputs)
% create the folder where the cropped images will be stored.

cd(CGONinputs.where2Store)
splitFiles=regexp(CGONinputs.path2GanNifti,filesep,'split');
convertedFolder=[splitFiles{end},'-','native'];
mkdir(convertedFolder); % creating the converted folder for storing
path2ConvFolder=[CGONinputs.where2Store,filesep,convertedFolder];


% load the nifti files.

cd(CGONinputs.path2GanNifti);
niftiFiles=dir('*.nii');
cropRange=44;
parfor lp=1:length(niftiFiles)
    imgVol=niftiread(niftiFiles(lp).name);
    xMax=344;
    yMax=344;
    zMax=127;
    newVol{lp}=imgVol(:,:,1:zMax);
end

cd(CGONinputs.path2OrgNifti)
orgNiftiFiles=dir('*.nii');
parfor lp=1:length(orgNiftiFiles)
    cd(CGONinputs.path2OrgNifti)
    NativeFileName=['PET-Nav-',orgNiftiFiles(lp).name]; 
    nativeVolumes=niftiread(orgNiftiFiles(lp).name);
    nativeVolumes(cropRange+1:xMax-cropRange,cropRange+1:yMax-cropRange,:)=newVol{lp};
    hdrInfo=niftiinfo(orgNiftiFiles(lp).name);
    hdrInfo.Description='GAN derived PET navigators';
    niftiwrite(nativeVolumes,NativeFileName,hdrInfo);
    disp(['Writing ',NativeFileName,'...']);
    movefile(NativeFileName,path2ConvFolder);
end


end