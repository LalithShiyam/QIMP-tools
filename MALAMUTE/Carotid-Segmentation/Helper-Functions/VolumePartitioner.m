function [ FV1,FV2 ] = VolumePartitioner( FinalVolume )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

FV1=zeros(size(FinalVolume));
conncomp=bwconncomp(FinalVolume);
[~,i4]=max(cellfun('size',conncomp.PixelIdxList,1));
FV1(conncomp.PixelIdxList{i4})=1;
conncomp.PixelIdxList{i4}=[];
FV2=zeros(size(FinalVolume));
[~,i4]=max(cellfun('size',conncomp.PixelIdxList,1));
FV2(conncomp.PixelIdxList{i4})=1;



end

