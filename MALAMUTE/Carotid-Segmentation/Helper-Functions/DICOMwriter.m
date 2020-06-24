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
VolumeToBeWritten=VolumeToBeWritten;%.*1000;

for lp=1:size(VolumeToBeWritten,3)
    metadata=dicominfo(FileNames(lp).name);
    if isBinary==true
        metadata.RescaleSlope=1;
    end
    srlno=metadata.InstanceNumber;
    x=uint16(VolumeToBeWritten(:,:,double(srlno)));
    dicomwrite(x,strcat(strcat(FileExtName,'_'),num2str(srlno),'.IMA'),metadata,'CreateMode','copy');
end

MoveString=strcat(FileExtName,'_','*');
movefile(MoveString,strcat(TempWorkingDir,filesep,FolderName));

end
