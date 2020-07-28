%-----------------------------------------------------------------------
% Job saved on 06-May-2017 01:14:52 by cfg_util (rev $Rev: 6460 $)
% spm SPM - SPM12 (6906)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------
function [matlabbatch] = Klif_dicom2niftiJob_job(JobDicom2Nifti)
matlabbatch{1}.spm.util.import.dicom.data = JobDicom2Nifti.inputFiles;
matlabbatch{1}.spm.util.import.dicom.root = 'flat';
matlabbatch{1}.spm.util.import.dicom.outdir = {JobDicom2Nifti.outputDir};
matlabbatch{1}.spm.util.import.dicom.protfilter = '.*';
matlabbatch{1}.spm.util.import.dicom.convopts.format = 'nii'; % change it to 'nii' for moco and 'img' for extracting parametric values
matlabbatch{1}.spm.util.import.dicom.convopts.icedims = 0;
spm('defaults', 'PET');
spm_jobman('initcfg');
spm_jobman('run',matlabbatch);
end 
    