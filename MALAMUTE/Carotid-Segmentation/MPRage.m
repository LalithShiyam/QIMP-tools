function MPRageIm = MPRage(MPRpath)
cd(MPRpath);

Images=dir;
Images=Images(arrayfun(@(x) x.name(1), Images) ~= '.'); 
if length(Images)==0
    Images = dir('*.dcm');
end
if length(Images)==0
    Images = dir('*');
    Images=Images(~[Images.isdir]); %removes the variables which do not have a name.

end


h = cProgress ( 0, 'Reading images'); 
for i=1:length(Images);
    cProgress(100.*(i/length(Images)),h)
    info = dicominfo(Images(i).name);
    im = dicomread(Images(i).name); im = double(im);
    if length(Images)==1
        MPRageIm=squeeze(im);
    else
        MPRageIm(:,:,info.InstanceNumber) = im;
    end
end
close(h)