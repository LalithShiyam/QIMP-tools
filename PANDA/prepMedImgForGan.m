%------------------------------------------------------------------------%
% This function converts images in medical imaging formats (Nifti or
% Analyze) to commercially usable formats (JPG, PNG)
% 
% Note: The code was developed internally to process DICOM images to GAN
% compatible 'png' or 'jpg' formats. Therefore, the code expects at
% somepoint the inherent matrix size of the pet system (344 x 344 x 127).
% Therefore, it is a limitation. But this can be easily overcome with some
% modification.
%
% Author : Lalith Kumar Shiyam Sundar
% Date   : 18 November, 2019
%
% Inputs: 
%       [1]PMIGinputs.path2MedImg: file path to the medical image.
%       [2]PMIGinputs.where2Store: file path to store the generated images.
%       [3]PMIGinputs.fileFormat: 'jpg' or 'png'.
%
% Outputs: Folder containing the converted images. 
%
% Usage: prepMedImgForGan(PMIGinputs);
%       
%------------------------------------------------------------------------%
%                           Program start
%------------------------------------------------------------------------%

function [] = prepMedImgForGan(PMIGinputs)

path2MedImg=PMIGinputs.path2MedImg;
where2Store=PMIGinputs.where2Store;
fileFormat=PMIGinputs.fileFormat;

% Create the folder to store the converted images.

splitFiles=regexp(path2MedImg,filesep,'split')
convertedFolder=[splitFiles{end},'-',fileFormat];
cd(where2Store)
mkdir(convertedFolder);
where2Store=[where2Store,filesep,convertedFolder];


% Find out the format of the medical images: Dicom, nifti or analyze.

medFormat=checkFileFormat(path2MedImg);
cd(path2MedImg)
switch medFormat
    case 'Analyze'
        medFiles=dir('*hdr')
        for lp=1:length(medFiles)
            medImg{lp}=analyze75read(medFiles(lp).name);
        end
    case 'Nifti'
        medFiles=dir('*nii')
        for lp=1:length(medFiles)
            medImg{lp}=niftiread(medFiles(lp).name);
        end
end

% converting the medical image (nifti or analyze) to jpg/png.

for olp=1:length(medImg)
    disp(['Processing ',medFiles(olp).name,'...']);
    tempImg=medImg{olp};
    [ganCmpImg]=makeImgGanCompatabile(tempImg,PMIGinputs);
    parfor lp=1:size(ganCmpImg,3)
        pngImg=mat2gray(ganCmpImg(:,:,lp)); 
        pngFileName=[medFiles(olp).name,'-',num2str(lp),'.',fileFormat];
        imwrite(pngImg,pngFileName)
        disp(['Writing slice number ',num2str(lp),'...']);
        movefile(pngFileName,where2Store)
        disp(['Moving ',pngFileName,' to ',where2Store])
    end
end

end

% Local function 

function [ganCmpImg]=makeImgGanCompatabile(imgVol)

% Hard-coded variables
    cropMargin=44;
    ganCmpImg=imgVol;
    xMax=size(imgVol,1);
    yMax=size(imgVol,2);
    ganCmpImg=ganCmpImg(cropMargin+1:(xMax-cropMargin),cropMargin+1:(yMax-cropMargin),:);    
end

