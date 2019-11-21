%------------------------------------------------------------------------%
% This function prepares the images for GAN training. The images are sorted
% into 3 folders: (1) train (60% of total images) (2) test (20% of total
% images) and (3) val (20% of total images). Also the images are combined
% into pairs by using the scripts of pix2pix (github).
%
% Author : Lalith Kumar Shiyam Sundar
% Date   : 21 November, 2019
%
% Inputs: 
%       [1]PDTinputs.pathOfSource: file path to the source images.
%       [2]PDTinputs.pathOfTarget: file path to the target images.
%       [3]PDTinputs.whereToStore: file path to store the generated images
%       [4]PDTinputs.sourceTargetNames: e.g: {'A','B'}; 
% Outputs: Folders needed for running the pix2pix program. 
%
% Usage: prepDataForTraining(PDTinputs);
%       
%------------------------------------------------------------------------%
%                           Program start
%------------------------------------------------------------------------%
function []=prepDataForTraining(PDTinputs)

% Hard-coded variables.
workingFolder=[PDTinputs.sourceTargetNames{1},'-',PDTinputs.sourceTargetNames{2}];
trainingFolder='train';
validationFolder='val';
testingFolder='test';


cd(PDTinputs.whereToStore) 
mkdir(workingFolder);
pathOfWorkingFolder=[PDTinputs.whereToStore,filesep,workingFolder];
splitFiles=regexp(PDTinputs.pathOfSource,filesep,'split')
sourceFolder=splitFiles{end};
cd(pathOfWorkingFolder)
mkdir(sourceFolder);
pathToSourceFolder=[pathOfWorkingFolder,filesep,sourceFolder];
splitFiles=regexp(PDTinputs.pathOfTarget,filesep,'split')
targetFolder=splitFiles{end};
cd(pathOfWorkingFolder)
mkdir(targetFolder);
pathToTargetFolder=[pathOfWorkingFolder,filesep,targetFolder];

% source folder split.

cd(pathToSourceFolder)
mkdir(trainingFolder);mkdir(testingFolder);mkdir(validationFolder);
sourceTrainingPath=[pathToSourceFolder,filesep,trainingFolder];
sourceTestingPath=[pathToSourceFolder,filesep,testingFolder];
sourceValidationPath=[pathToSourceFolder,filesep,validationFolder];


% Move 60 percent of source images to the training folder, 20 percent to
% testing and validation folder.

cd(PDTinputs.pathOfSource);
totalFiles=dir; totalFiles=totalFiles(arrayfun(@(x) x.name(1), totalFiles) ~= '.'); % read the file names in the folder
numTotalFiles=length(totalFiles);
trainNumber=round(0.6*numTotalFiles);
testNumber=round(0.2*numTotalFiles);
valNumber=round(0.2*numTotalFiles);
parfor lp=1:trainNumber
   movefile(totalFiles(lp).name,sourceTrainingPath);
end
totalFiles=dir; totalFiles=totalFiles(arrayfun(@(x) x.name(1), totalFiles) ~= '.'); % read the file names in the folder
parfor lp=1:testNumber
    movefile(totalFiles(lp).name,sourceTestingPath);
end
totalFiles=dir; totalFiles=totalFiles(arrayfun(@(x) x.name(1), totalFiles) ~= '.'); % read the file names in the folder
parfor lp=1:valNumber
    movefile(totalFiles(lp).name,sourceValidationPath);
end

% target folder split.

cd(pathToTargetFolder)
mkdir(trainingFolder);mkdir(testingFolder);mkdir(validationFolder);
targetTrainingPath=[pathToTargetFolder,filesep,trainingFolder];
targetTestingPath=[pathToTargetFolder,filesep,testingFolder];
targetValidationPath=[pathToTargetFolder,filesep,validationFolder];

% Move 60 percent of target images to the training folder, 20 percent to
% the testing and validation folder.

cd(PDTinputs.pathOfTarget)
totalFiles=dir; totalFiles=totalFiles(arrayfun(@(x) x.name(1), totalFiles) ~= '.'); % read the file names in the folder
numTotalFiles=length(totalFiles);
trainNumber=round(0.6*numTotalFiles);
testNumber=round(0.2*numTotalFiles);
valNumber=round(0.2*numTotalFiles);

parfor lp=1:trainNumber
   movefile(totalFiles(lp).name,targetTrainingPath);
end
totalFiles=dir; totalFiles=totalFiles(arrayfun(@(x) x.name(1), totalFiles) ~= '.'); % read the file names in the folder
parfor lp=1:testNumber
    movefile(totalFiles(lp).name,targetTestingPath);
end
totalFiles=dir; totalFiles=totalFiles(arrayfun(@(x) x.name(1), totalFiles) ~= '.'); % read the file names in the folder
parfor lp=1:valNumber
    movefile(totalFiles(lp).name,targetValidationPath);
end


end