%% Carney's bilinear scaling
% This function converts a given low-dose CT image to CT-umap, by using carney's transformation technique
% Literature: (Carney et.al, 2006, Transforming CT images for attenuation correction in PET/CT)
% Author: Lalith Kumar Shiyam Sundar, M.Sc. 
% Date: Feb 15, 2018
% Quantitative Imaging and Medical Physics, Medical University of Vienna.
% Inputs: pathOfCTdicom: Path of the CT dicom series (low-dose)
%         CTimgToScale: The low-dose CT volume, which needs to be converted
%         to a u-map.
% Outputs: CTuMap: The CT attenuation map, scaled using carney's functions.
% 
% Usage: pathOfCTdicom= 'users/desktop/AC_CT_XYZ';
%        CTimgToScale=CTvolume; % matlab 3D matrix
%        CTuMap=carneyBilinearScaling(pathOfCTdicom,CTimgToScale); % press
%        enter.

function CTuMap=carneyBilinearScaling(pathOfCTdicom,CTimgToScale)

% Parameters for Bilinear scaling, choose a, b and BP.
cd(pathOfCTdicom);
CTfiles=dir;
CTfiles=CTfiles(arrayfun(@(x) x.name(1), CTfiles) ~= '.'); % reading the files inside the folder.
CTdcmInfo=dicominfo(CTfiles(1).name);
TubeVoltage=CTdcmInfo.KVP;
scalingFactorSoftTissue=9.6e-005; % obtained from siemens LUT parameters of PET/CT TPTV.
switch TubeVoltage % these values are obtained from the literature (Carney et.al, 2006, Transforming CT images for attenuation correction in PET/CT)
    case 80
    a  = 3.64e-005;
    b  = 6.26e-002;
    BP = 1050; % (HU+1000) based on scaled components (0 to 4000)
    case 100
    a  = 4.43e-005;
    b  = 5.44e-002;
    BP = 1052; 
    case 110
    a  = 4.92e-005;
    b  = 4.88e-002;
    BP = 1043; 
    case 120
    a  = 5.10e-005;
    b  = 4.71e-002;
    BP = 1047;
    case 130
    a  = 5.51e-005;
    b  = 4.24e-002;
    BP = 1037; 
    case 140
    a  = 5.64e-005;
    b  = 4.08e-002;
    BP = 1030; 
end
 
% Scaling the CT using the bilinear coefficients.
disp(['Low-dose CT to u-map conversion parameters: ',num2str(scalingFactorSoftTissue),', ',num2str(a),', ',num2str(b),', ',num2str(BP),'!']);
CTbelowBPmask=CTimgToScale<=(BP+23); % Siemens recommends using 1070 for segmenting bone and soft-tissue, this value can be seen in the e7 tools.
CTbelowBP=CTimgToScale.*(CTbelowBPmask);
CTbelowBPto511KeV=(scalingFactorSoftTissue).*(CTbelowBP).*CTbelowBPmask; % Values in linear attenuation coefficients, units cm-1
CTaboveBPMask=CTimgToScale>(BP+23);
CTaboveBP=CTimgToScale.*(CTaboveBPMask);
CTaboveBPto511KeV=((a.*(CTaboveBP))+b).*CTaboveBPMask; % values in linear attenuation coefficients, units cm-1
CTKVPto511KeV=CTbelowBPto511KeV+CTaboveBPto511KeV;
CTKVPto511KeV=(10000.*CTKVPto511KeV); % specially scaled for siemens mMR PET reconstruction. Artificial attenuation maps need to be in the units of 10k cm-1.
CTKVPto511KeV(CTKVPto511KeV<0)=0;
CTuMap=CTKVPto511KeV;
end
