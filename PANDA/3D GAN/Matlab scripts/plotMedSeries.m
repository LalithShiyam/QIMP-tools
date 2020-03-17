function [] = plotMedSeries(pathOfNiftiSeries)

% read the files in the folder

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

% create a plot 

figure('units','normalized','outerposition',[0 0 1 1])
for lp=1:(length(img)-1)
    subplot(6,6,lp)
    midSlice=size(img{lp},3);
    imshow(img{lp}(:,:,round(midSlice/2)),[])
    zoom(3) 
    title(['PET-Frame: ',num2str(lp)])
end

end