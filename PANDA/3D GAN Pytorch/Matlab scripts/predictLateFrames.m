% This function predicts late frames from early frames using the model
% files obtained from training.
%
% Author : Lalith Kumar Shiyam Sundar
% Date   : 12 December, 2019
%
% Inputs: 
%       [1]PLFinputs.pathOfEarlyFrame: file path to the early frame
%       [2]PLFinputs.where2Store: file path to store the predicted image.
%       [3]PLFinputs.pathToModel: file path of the deep learning model
%       
% Outputs: Folder containing the predicted images. 
%
% Usage: predictLateFrames(PLFinputs);
%       
%------------------------------------------------------------------------%
%                               Program start
%------------------------------------------------------------------------%
function []=predictLateFrames(PLFinputs)

prefixPowShellScript=['!powershell -executionPolicy bypass ". ']; % with dot sourcing.

pathOfEarlyFrame=PLFinputs.pathOfEarlyFrame;
pathToModel=PLFinputs.pathToModel;
where2Store=PLFinputs.where2Store;
splitFiles=regexp(pathOfEarlyFrame,filesep,'split');
convertedFile=['GAN-',splitFiles{end}];
where2Store=[where2Store,filesep,convertedFile];
pathOfPythonGan=what('3DGanPython');
cd(pathOfPythonGan.path);
tempPath=pathOfPythonGan.path;
tempPath = regexprep(tempPath, ' ', '\\ ');
string2Run=['cd ',tempPath]; 
system(string2Run);
stringToRun=['python predict_single_image.py  --image ','"',pathOfEarlyFrame,'"',' --result ','"',where2Store,'"',' --gen_weights ','"',pathToModel,'"'];
[stat,cmdOut]=system(stringToRun,'-echo')

end

