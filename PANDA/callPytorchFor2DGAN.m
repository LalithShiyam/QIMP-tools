%------------------------------------------------------------------------%
% This function calls the pytorch scripts for creating image pairs
% 
%                        * Important Note *
%
% The PyTorch scripts are taken from Jun-Yan Zhu repository
% https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
% Kindly cite his work if you have used this script, citations are provided
% in his repository
%
% Author : Lalith Kumar Shiyam Sundar
% Date   : 1 December, 2019
%
% Inputs: 
%       [1]CPGinputs.pathOfPytorchFolder: file path to the Jun-Yan Zhu's scripts.
%       [2]CPGinputs.pathOfSourceImage: file path to the source images.
%       [3]CPGinputs.pathOfTargetImage: file path to the target Image
%       [4]CPGinputs.where2Store: file path to store the image pairs

% Outputs: Image-pairs needed for running the pix2pix program of Jun-Yan Zhu. 
%
% Usage: callPytorchFor2DGAN(CPGinputs);
%       
%------------------------------------------------------------------------%
%                           Program start
%------------------------------------------------------------------------%

function [] = callPytorchFor2DGAN(CPGinputs)

% Hard-coded variables 

nameOfShellFile='call-Pytorch.command';

% Local variable move

pathOfPytorch=CPGinputs.pathOfPytorchFolder;
pathOfSourceImg=CPGinputs.pathOfSourceImage;
pathOfTrgtImg=CPGinputs.pathOfTargetImage;
pathOfStorage=CPGinputs.where2Store;

% create a shell file for running Jun-Yan Zhu's scripts
cd(pathOfStorage)
defaultHeader='#! /bin/bash'
stringToGo2PytorchFolder=['cd ',pathOfPytorch];
path2CombineAandB=[pathOfPytorch,filesep,'datasets/combine_A_and_B.py'];
string2Run=['python ',path2CombineAandB,' --fold_A ',pathOfSourceImg,' --fold_B ',pathOfTrgtImg,' --fold_AB ',pathOfStorage];
strings2Write{1}=defaultHeader;
strings2Write{2}=stringToGo2PytorchFolder;
strings2Write{3}=string2Run;
fid=fopen(nameOfShellFile,'wt');
fprintf(fid,'%s\n',strings2Write{:});
fclose(fid);
path2PytorchShell=fileparts(which("call-Pytorch.command"))
path2PytorchShell=[path2PytorchShell,filesep,nameOfShellFile];
string2Run=['chmod u+x ',path2PytorchShell];
system(string2Run)
PATH = getenv('PATH');
setenv('PATH', [PATH ':/usr/local/desiredpath']);
disp(['Creating image pairs from ',pathOfSourceImg,' and ',pathOfTrgtImg]);
end


