%% This function will calculate 3D gradient images from a given series of 
%  PET images.
% 
% 
%  Author: Lalith Kumar Shiyam Sundar, M.Sc.
%  Date: March 15, 2020
%  Quantitative Imaging and Medical Physics, Medical University of Vienna.
%  
%  Incase of errors or troubles, contact: lalith.shiyamsundar@meduniwien.ac.at 
%  
%  Usage:
%  
%  pathToNiftiPET='/Users/lalith/Documents/PANDA';
%  where2Store='/Users/lalith/Documents'; 
%  genGradImg(pathToNiftiPET,where2Store);
%  'Gradient images stored in /Users/lalith/Documents/PANDA-gradient-images'
%
%  
%-------------------------------------------------------------------------%
%                               Program start
%-------------------------------------------------------------------------%
function [] = genGradImg(pathToNiftiPET,where2Store)

% hard-coded variables 
 
normalizationFactor=44; % gradient field normalisation 1/44 for sobel

cd(where2Store)
mkdir('PANDA-gradient-images');
where2Store=[where2Store,filesep,'PANDA-gradient-images'];
cd(pathToNiftiPET)
niftyFiles=dir('*nii');
parfor lp=1:length(niftyFiles)
names{lp}=niftyFiles(lp).name;
end
sortedNames=natsort(names);

for lp=1:length(sortedNames)
cd(pathToNiftiPET);
disp(['Calculating gradient image for ',sortedNames{lp},'...'])
img{lp}=spm_read_vols(spm_vol((sortedNames{lp})));
imgInfo=spm_vol(sortedNames{lp});
gradImg{lp}=imgradient3(img{lp})./normalizationFactor;
gradImg{lp}=gradImg{lp}./(mean(gradImg{lp}(:)));
gradImgName=[sortedNames{lp}];
gradImgName=['Gradient-image-',sortedNames{lp}];
imgInfo.fname=gradImgName;
grads=(gradImg{lp});
spm_write_vol(imgInfo,grads);
movefile(gradImgName,where2Store)
end
end