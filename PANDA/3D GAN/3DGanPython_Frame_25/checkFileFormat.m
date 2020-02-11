%-------------------------------------------------------------------------%
% This is a function which checks the format of the medical images in a given
% folder. 
% Usage:
%   fileFormat = checkFileFormat(filePath)  
%   Output: "fileFormat" is a string, can either be "DICOM", "NIFTI" or "Analyze"
%   Input: "filePath" is the folder path, where the images are stored.
% Example: 
%   fileFormat = checkFileFormat('/Users/work/Documents/Git sync/CT to
%   MR/Scripts/Helper functions');
%                                   or
%   filePath='/Users/work/Documents/Git sync/CT to MR/Scripts/Helper
%   functions'; 
%   fileFormat = checkFileFormat(filePath);
%
% Author - Lalith Kumar Shiyam Sundar, M.Sc.
% Date   - 30-12-2017
% Mail   - lalith.shiyamsundar@meduniwien.ac.at
%
%-------------------------------------------------------------------------%
%                           Program Start 
%-------------------------------------------------------------------------%

function fileFormat = checkFileFormat(pathOfTheImages)

%% Input checks

if isempty(pathOfTheImages)
    error('Input parameter is empty, please specify the file path!')
end

if ~ischar(pathOfTheImages)
    error('Input parameter is probably not a filepath!');
end

%% Go to the file path

cd(pathOfTheImages);

checkIfAnalyze=dir('*.img');
checkIfNifti=dir('*.nii');
if isempty(checkIfAnalyze) && isempty(checkIfNifti)
    fNames=dir;
    fNames=fNames(arrayfun(@(x) x.name(1), fNames) ~= '.'); % read volunteer folders
    CheckIfDicom=dicominfo(fNames(1).name); % if dicominfo can read the file, probably its dicom, lazy logic.
end

if ~isempty(checkIfAnalyze)
    fileFormat='Analyze';
    disp('Analyze files found...')
else
    if ~isempty(checkIfNifti)
        fileFormat='Nifti';
        disp('Nifti files found...')
    else
        if ~isempty(CheckIfDicom)
            fileFormat='Dicom';
            disp('Dicom files found...')
        end
    end
end

end



