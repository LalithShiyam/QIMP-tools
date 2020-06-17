%Getting the filenames in a folder
function[]= DICOMwriter(FolderName,VolumeToBeWritten,FileExtName,WhereToStore,isBinary)

FileNames=dir(cd);
FileNames=FileNames(~[FileNames.isdir]);
ActualWorkingDir=cd;
FileNames=FileNames(1:end);

cd(WhereToStore) % going back a folder.
TempWorkingDir=cd;
mkdir(cd,FolderName); %Making a folder - segmented internal carotid artery
cd(ActualWorkingDir)    % Going to the actual working directory where the files are stored
%VolumeToBeWritten=VolumeToBeWritten;%.*1000;
SZ=size(VolumeToBeWritten,3);
End1=round(SZ*0.5);
Beg2=End1+1;

parfor  lp=1:End1
        metadata=dicominfo(FileNames(lp).name);
    if  isBinary==true
        metadata.RescaleSlope=1;
    end
    srlno=metadata.InstanceNumber;
    metadata.SliceThickness=0.5;
    metadata.Rows=511;
    metadata.Columns=447;
    %metadata.PixelSpacing=0.5\0.5;
    x=uint16(VolumeToBeWritten(:,:,lp));
    metadata.RadiopharmaceuticalInformationSequence.Item_1.RadionuclideTotalDose=2002819;
   %dicomwrite(x,strcat(strcat(FileExtName,'_'),num2str(srlno),'.IMA'));
    dicomwrite(x,strcat(strcat(FileExtName,'_'),num2str(lp),'.IMA'),metadata,'CreateMode','copy');
end

parfor  lp=Beg2:SZ
        metadata=dicominfo(FileNames(lp-192).name);
    if  isBinary==true
        metadata.RescaleSlope=1;
    end
    srlno=metadata.InstanceNumber;
    metadata.SliceThickness=0.5;
    metadata.Rows=511;
    metadata.Columns=447;
    %metadata.PixelSpacing=0.5\0.5;
    metadata.InstanceNumber=(srlno+192);
    x=uint16(VolumeToBeWritten(:,:,lp));
    metadata.RadiopharmaceuticalInformationSequence.Item_1.RadionuclideTotalDose=2002819;
   %dicomwrite(x,strcat(strcat(FileExtName,'_'),num2str(srlno),'.IMA'));
    dicomwrite(x,strcat(strcat(FileExtName,'_'),num2str(lp),'.IMA'),metadata,'CreateMode','copy');
end

MoveString=strcat(FileExtName,'_','*');
movefile(MoveString,strcat(TempWorkingDir,filesep,FolderName));

end
