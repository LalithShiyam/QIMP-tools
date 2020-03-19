function [] = plotMedSeries(pathOfNiftiSeries)

%% read the files in the folder
rows=6;
columns=6;
zoomFactor=3.5;
cd(pathOfNiftiSeries)
niftiFiles=dir('*nii');
parfor lp=1:length(niftiFiles)
    files2Sort{lp}=niftiFiles(lp).name;
end
sortedNiftiFiles=natsort(files2Sort);

% Load the medical images 

parfor lp=1:length(sortedNiftiFiles)
     disp(['Reading ',sortedNiftiFiles{lp}])
    img{lp}=niftiread(sortedNiftiFiles{lp});
end
clc 

%% create a plot 

figure('units','normalized','outerposition',[0 0 1 1])
%timeStamps=getPETmidTime;
for lp=1:(length(img)-1)
    subaxis(rows,columns,lp, 'sh', 0, 'sv', 0.001, 'padding', 0, 'margin', 0);
    midSlice=size(img{lp},3);
    imshow(img{lp}(:,:,round(midSlice/2)),[])
    zoom(zoomFactor) 
    ylabel(['Frame: ',num2str((lp))],'fontweight','bold','fontsize',15)
end
figure('units','normalized','outerposition',[0 0 1 1])
for lp=1:(length(img)-1)
    subaxis(rows,columns,lp,'sh',0,'sv',0.001,'padding',0,'margin',0);
    dummyImg=img{lp};
    voxValues=dummyImg(:);
    newVoxVal=voxValues(voxValues<2);
    histogram(newVoxVal)
    ylabel(['Histogram for frame: ',num2str(lp)],'fontweight','bold','fontsize',5)
end