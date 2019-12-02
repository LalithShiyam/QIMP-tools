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

% Local variable move

pathOfPytorch=CPGinputs.pathOfPytorchFolder;
pathOfSourceImg=CPGinputs.pathOfSourceImage;
pathOfTrgtImg=CPGinputs.pathOfTargetImage;
pathOfStorage=CPGinputs.where2Store;

% change the current directory 

stringToGo2PytorchFolder=['cd ',pathOfPytorch];
status=system(stringToGo2PytorchFolder);

% Run Jun-Yan Zhu's scripts for creating image pairs.

string2Run=['python datasets/combine_A_and_B.py --fold_A ',pathOfSourceImg,' --fold_B ',pathOfTrgtImg,' --fold_AB ',pathOfStorage];
status=system(string2Run);
disp(['Creating image pairs from ',pathOfSourceImg,' and ',pathOfTrgtImg]);
end


end