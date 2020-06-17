%% Purpose
% This is the core program which is responsible for the automatic petrous
% and cervical segmentation of the internal carotid arteries. The
% segmentation is based on Gibo et.al


%% References
% Gibo H, Lenkey C and Rhoton AL. Microsurgical anat- omy of the supraclinoid portion of the internal carotid artery. J Neurosurg 1981; 55: 560â€“574.  


%% Author information
% Lalith Kumar Shiyam Sundar, 
% Quantitative Imaging and Medical Physics, Medical University of Vienna


%% Adapted 06/2020
% core program for extracting a simple petrous mask out of a contrast
% enhanced T1-brain-MR by Matthias


%% Inputs
% coreSegInputs.pathOfMR - Physical path of the contrast enhanced
% T1-brain-MR in DICOM format
% This will be provided by the "wrapperForSegmentation.m"

%% Logic of the program
% Setting Hard-coded variables to play with
% 0. Preliminaries
%    Reading in the DICOM
%    Variably interpolation for smoother final mask
% 1. Carotid vascular segmentation
%    Extracting only the highest pixel values
%    Defining a Seeding slice where the carotides will surely be
%    Fínding the SeedPoints
%    Region Growing around the SeedPoints to obtain the whole arteries
% 2. Intensity Feature Analysis
%    Calculating the normalised mean intensity for the found arteries
% 3. Cropping the boarders and Morphometric Analysis
%    Cropping of the borders to prevents wrap around artefacts
%    Calculating the major and minor axis length for the found arteries
%    Multyply the MajorMinor analysis with the normalised mean intensity
%    Searching for the Peaks of this curve and sorting the peaks
% 4. Getting the petrous slice
%    Using the maxPeak of the Morphometric Analysis as start point
%    Cropping out the predefined number of slices around this starting point
% 5. Output
%    Option to resize the image if interpolated
%    Output is written in a DICOM format
%    Figures for quick checking:
%    resulting Arteries, Morphometric Analysis curve, cropped petrous mask


%% Program start

function [] = coreSegmentationMP ( coreSegInputs )

%% Hard-coded variables: change it according to your needs.

upperSliceLimit = 192; % total amount is 192, If set lower, the SliceSelecter has to be adapted
quantileVal     = 0.9997; % Threshold value for the preselection of the highest voxel values (mainly in the arteries), the lower this value, the higher the SeedingThrashold must be
SliceSelecter   = 0.25; % partition of the total brain Volume, at which the Seeding slice is selected
SeedingThreshold= 0.85; % Threshold value for the Seeding points; the lower the higher the risk for the code to break, depends on the quantileVal
GrowingThreshold= 0.10; %Threshold value for the region growing algorithm, the higher the more will be included, depends on the quantileVal, Standard 0.05
wrapPrevention  = 50; % Number of pixels that will be cropped from the border to prevent wrap-around artefacts
PetrousSize    = 16; % Number of slices in the petrous mask, double if interpolation is used

%% Logic start


%% 0. Preliminaries

% Read in the DICOM images
orgVolume=MPRage(coreSegInputs.pathOfMR); % external program (Helper functions)
orgVolume(:,:,upperSliceLimit:end)=zeros; % nulling the values beyond this point.

% % cubic Interpolation for smoother Petrous mask in the end
% disp('Applying Cubic interpolation on the dataset');
% orgVolume=interp3(orgVolume,'cubic');


%% 1. Carotid vassculature segmentation

% Inputs
segCVinputs.imgVolume=orgVolume;
segCVinputs.qVal=quantileVal;
segCVinputs.SliceSel=SliceSelecter;
segCVinputs.SeedThresh=SeedingThreshold;
segCVinputs.GrowThresh=GrowingThreshold;

% Carotid vasculature segmentation program, see below
[segCVoutputs]=segCV(segCVinputs); 


%% 2. Intensity Feature Analysis

% Inputs
IFAinputs.orgVolume=orgVolume;
IFAinputs.bVol=segCVoutputs.bVol;

% Intensity Feature Analysis Program, see below
[IFAoutputs]=intensityFeatureAnalysis(IFAinputs);


%% 3. Cropping Boarders and Morphometric Analysis

% Inputs
MMAinputs.bVol=cropBorders(segCVoutputs.bVol,wrapPrevention); %cropBorders program, see below
MMAinputs.curveOfInterest=IFAoutputs.curveOfInterest;

% Morphometric Analysis Program, see below
[MMAoutputs]=MMAnalysis(MMAinputs);


%% 4. Getting petrous slices

% Inputs
PSinputs.bVol=MMAoutputs.bVol;
PSinputs.SXAxis=MMAoutputs.sortedXAxis;
PSinputs.numSlices=PetrousSize;

% Petrous slicer, see below
[PSoutputs]=PSlicer(PSinputs);


%% 5. Creating Output

% Inputs
PetrousMask=PSoutputs.PetrousMask;

% % Resizing
% Mask1=imresizen(Mask1,[0.5,0.5,0.5],'cubic');

% figure for quick checking
figure, VolumeVisualizer(PetrousMask,'r');

% Writing the resulting petrous mask into a DICOM, external program (Helper
% functions) (If interpolation is used, use DICOMwriterMP)
whereToStore=coreSegInputs.path2StoreSeg;
DICOMwriter([coreSegInputs.patientCode,'_Petrous_Mask'],PetrousMask,coreSegInputs.patientCode,whereToStore,1);
% DICOMwriterMP([coreSegInputs.patientCode,'_',Name1],PetrousMask,Name1,whereToStore,1);
end




%% Local programs


%% 1. For Segmenting the entire carotid vasculature.

function [ segCVoutputs ] = segCV ( segCVinputs )

% passing inputs on to local variables.
orgVol=segCVinputs.imgVolume;
qVal=segCVinputs.qVal;
SliceSel=segCVinputs.SliceSel;
SeedThresh=segCVinputs.SeedThresh;
GrowThresh=segCVinputs.GrowThresh;

% preliminary segmentation, just taking the highest pixel values to create
% a basic first mask
qMask=(orgVol>quantile(orgVol(:),qVal));

% Selecting a particular starting slice where the ICA is bright for automatic
% seeded region growing algorithm
orgVol2D=orgVol(:,:,round(size(orgVol,3)*SliceSel));
orgVol2D=double(orgVol2D); 

% Threshold for getting the seed-points
threshold=round(SeedThresh*(max(orgVol2D(:))));

% Finding the SeedPoints
BM=orgVol2D>threshold;
skelImg=bwmorph(BM,'skel',inf);
seedImg=bwmorph(skelImg,'shrink');
[r,c]=find(seedImg==1);

% Getting the automatically selected seeds as matrix
seedMatrix=[r,c];
seedMatrix=[seedMatrix,round(size(orgVol,3)*SliceSel)*ones(size(seedMatrix,1),1)];

% Preliminaries for the region Growing algorithm
bVol=zeros(size(orgVol));       
thresVal = double((max(qMask(:))-min(qMask(:))))*GrowThresh;

% Region growing algorithm, external program (Helper functions)
for lp=1:size(seedMatrix,1)
    [~,maskOfInt]=regionGrowing(qMask,seedMatrix(lp,:),thresVal);
    bVol=bVol+maskOfInt;
end

% Despacle, external program (Helper functions)
bVol=(bVol>=1);
[V1,V2]=VolumePartitioner(bVol);
clear bVol
bVol=V1+V2;
bVol=(bVol>=1);
segCVoutputs.bVol=bVol;
end
    

%% 2. Intensity Feature Analysis

function [ IFAoutputs ] = intensityFeatureAnalysis ( IFAinputs )

% Passing inputs on to local variables.
orgVol=IFAinputs.orgVolume;
bVol=IFAinputs.bVol;
gVol=orgVol.*bVol;
endPt=size(gVol,3);

% Initialisation for optimisation purposes
NumberOfObjects=zeros(1,size(gVol,3));
MeanIntensity=zeros(1,size(gVol,3));
NormalisedMeanIntensity=zeros(1,size(gVol,3));
MInonzero=zeros(1,size(gVol,3));
NMInonzero=zeros(1,size(gVol,3));

% Calculating the Mean intensity and the normalised mean intensity.
for lp=1:endPt
    MeanIntensity(lp)=mean2(gVol(:,:,lp));
    MInonzero(lp)=mean(nonzeros(gVol(:,:,lp)));
    [~,NumberOfObjects(lp)]=bwlabel(bVol(:,:,lp));
    NormalisedMeanIntensity(lp)=MeanIntensity(lp)./NumberOfObjects(lp); % Mean intensity normalised to the number of objects in each slice.
    NMInonzero(lp)=MInonzero(lp)./NumberOfObjects(lp);
end

% Getting only the NormalisedMeanIntensity
curveOfInterest=NormalisedMeanIntensity;
IFAoutputs.curveOfInterest=curveOfInterest;
end


%% 3.1 Cropping the volume to escape wrap-around artifacts.


function [ croppedVolume ] = cropBorders ( bVol,wrapPrevention )

% passing inputs on to local variables
cropFromBorder=wrapPrevention; % Number of pixels
croppedVolume=zeros(size(bVol));
mask=zeros(size(bVol));

% cropping from all sides
for lp=1:size(bVol,3)
    maskZero=ones(size(bVol(:,:,lp)));
     maskZero(1:cropFromBorder,:)=0;
     maskZero(:,1:cropFromBorder)=0;
     maskZero(:,(size(maskZero,2)-cropFromBorder):end)=0;
     maskZero((size(maskZero,1)-cropFromBorder):end,:)=0;
     mask(:,:,lp)=maskZero;
     croppedVolume(:,:,lp)=mask(:,:,lp).*bVol(:,:,lp);
end
end


%% 3.2 Morphometric Analysis

function [ MMAoutputs ] = MMAnalysis ( MMAnalysisInputs )

% passing inputs on to local variables
curveOfInterest=MMAnalysisInputs.curveOfInterest;
bVol=MMAnalysisInputs.bVol;

% Preliminaries
majorMinor=zeros(size(curveOfInterest));
[~,endInfo]=StartEndImgInfo(bVol);

% Major/Minor axis quantification for every object in each slice.
for lp=1:endInfo
    G=bwlabel(bVol(:,:,lp));
    FillArea=regionprops(G,'FilledArea');
    threshVF=double(round(max([FillArea.FilledArea])/5));   
    if  isempty(threshVF);
        threshVF=1;
    end
    
    BW2 = bwareaopen(bVol(:,:,lp),threshVF);
    if nnz(BW2)==0
        majorMinor(lp)=0;
    else
        [m1,~]=VolumePartitioner(BW2);
        Stats=regionprops(m1,'MajorAxisLength','MinorAxisLength');
        majorMinor(lp)=Stats.MajorAxisLength/Stats.MinorAxisLength;
    end
end

% Smoothing the resulting curves
smoothMajorMinor=sgolayfilt(majorMinor,4,15);
smoothMajorMinor(endInfo+1:size(bVol,3))=0;
smoothCOI=sgolayfilt(curveOfInterest,4,7);
smoothCOI(isnan(smoothCOI))=0;

% Morphometric analysis
curveOfInterest=smoothCOI.*smoothMajorMinor;
[Peaks,XAxis]=findpeaks(curveOfInterest);
[~,SortedIdx]=sort(Peaks,'descend');
sortedXAxis=XAxis(SortedIdx);
sortedXAxis=sortedXAxis(1:4);

% passing results on to the outputs
MMAoutputs.curveOfInterest=curveOfInterest;
MMAoutputs.bVol=bVol;
MMAoutputs.sortedXAxis=sortedXAxis;

% figures for quick checking
figure, imshow3D(MMAoutputs.bVol);
figure, plot(MMAoutputs.curveOfInterest);

end


%% 4. Petrous Slicer

function [ PSoutputs ] = PSlicer ( PSinputs )

% passing inputs on to local variables
bVol=PSinputs.bVol;
SXAxis=PSinputs.SXAxis;
numSlices=PSinputs.numSlices;

% Preliminaries
croppedVol=zeros(size(bVol));
endSlices=round(numSlices*0.5);
startSlices=(numSlices-endSlices);
TargetValue=SXAxis(1);
startVal=(TargetValue-startSlices);
endVal=(TargetValue+endSlices);

% Cropping the petrous slice out of the arteries
for lp=startVal:endVal
    croppedVol(:,:,lp)=bVol(:,:,lp);
end

% Passing the resulting Petrous mask on the the output
PSoutputs.PetrousMask=croppedVol;
end

