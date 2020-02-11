     %-----------------------------------------------------------------------
% Job saved on 08-May-2017 17:13:37 by cfg_util (rev $Rev: 6460 $)
% spm SPM - SPM12 (6906)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------
function [matlabbatch] = Coregistration_job(CoregInputs)
matlabbatch{1}.spm.spatial.coreg.estwrite.ref = {CoregInputs.RefImgPath};
matlabbatch{1}.spm.spatial.coreg.estwrite.source = {CoregInputs.SourceImgPath};
matlabbatch{1}.spm.spatial.coreg.estwrite.other = CoregInputs.MaskImgPath;
matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.cost_fun = 'nmi';
matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.sep = [4 2];
matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.fwhm = [7 7];
matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.interp = CoregInputs.Interp; % o - nearest neighbour, 1-trilinear, 2-2nd degree polynom, 3- 3rd deg polynomial . . . 7-7th deg polynomial.
matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.wrap = [0 0 0];
matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.mask = 0;
matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.prefix = CoregInputs.Prefix;
spm('defaults', 'PET');
spm_jobman('initcfg');
spm_jobman('run',matlabbatch);