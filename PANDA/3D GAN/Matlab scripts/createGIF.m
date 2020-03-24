%------------------------------------------------------------------------%
% This function creates an animated GIF file, based on the stacking of
% input images
%                       * Important note *
%
% The function expects a time-series of images which can be sorted. A
% standardised naming convention is expected, for example:
% A01.png,A02.png...A10.png etc.,
% 
%
% Author : Lalith Kumar Shiyam Sundar
% Date   : 3 December, 2019
%
% Inputs: 
%       [1]CGinputs.pathOfImageSeries: file path to the image series.
%       [2]CGinputs.where2Store: file path where the GIF file needs to be stored
%       [3]CGinputs.fileName: intented GIF file name - you need to add
%       ".gif" at the end
%
% Outputs: A GIF animation of the image-series.
%
% Usage: createGIF(CGinputs);
%       
%------------------------------------------------------------------------%
%                           Program start
%------------------------------------------------------------------------%


function []=createGIF(CGinputs)

% Move to local variables
pathOfImgSeries=CGinputs.pathOfImageSeries;
where2Store=CGinputs.where2Store;
GifFileName=CGinputs.fileName;

% read all image-series

cd(pathOfImgSeries)
imgFiles=dir('*jpg*');
if isempty(imgFiles)
    imgFiles=dir('*png*');
end

if isempty(imgFiles)
    imgFiles=dir('*tif*');
end
parfor lp=1:length(imgFiles)
    imgStack{lp}=imread(imgFiles(lp).name);
end

cd(where2Store)
h = figure;
axis tight manual % this ensures that getframe() returns a consistent size

for lp = 1:length(imgStack)
    % Draw plot for y = x.^n
    imshow(imgStack{lp},[]);
    title(imgFiles(lp).name)
    drawnow 
      % Capture the plot as an image 
      frame = getframe(h); 
      im = frame2im(frame); 
      [imind,cm] = rgb2ind(im,256); 
      % Write to the GIF File 
      if lp == 1 
          imwrite(imind,cm,GifFileName,'gif', 'DelayTime',1,'Loopcount',inf); 
      else 
          imwrite(imind,cm,GifFileName,'gif','DelayTime',1,'WriteMode','append'); 
      end 
end
close all
  
end