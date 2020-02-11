%% Author information
% 
%  Lalith Kumar Shiyam Sundar, M.Sc.
%  Quantitative Imaging and Medical Physics, Medical University of Vienna
%  Date: 27.11.2017, Wien
% 
% 
% A tiny helper program to convert DICOM files in a given folder

function []=convertDicomtoNii(pathOfDicom,whereToStore)
    cd(pathOfDicom)
    fileNames=dir;
       fileNames=fileNames(arrayfun(@(x) x.name(1),fileNames) ~= '.');
       for ilp=1:length(fileNames)
           dcmStrings{ilp,:}=[pwd,filesep,fileNames(ilp).name];
       end
    JobDicom2Nifti.inputFiles=dcmStrings; % preparing the input for personalised SPM_jobman for converting the input dicom files to a single nifti file.
    JobDicom2Nifti.outputDir=whereToStore; % Output directory for the converted NIFTI volumes.
    Klif_dicom2niftiJob_job(JobDicom2Nifti); % Klif_dicom2niftiJob_job is the script which does the nifti to dicom conversion.
end
