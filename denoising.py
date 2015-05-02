from nilearn.image import high_variance_confounds
import nibabel as nb
from nilearn.input_data import NiftiMasker
import numpy as np
from scipy.special import legendre
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd

df=pd.read_csv('/scr/ilz3/myelinconnect/subjects.csv', header=0)
subjects=list(df['DB'])
sessions=['rest1_1']#, 'rest1_2', 'rest2_1', 'rest2_2']

for subject in subjects:
    for session in sessions:
        
        print 'running '+subject+' '+session
        
        base_dir='/scr/ilz3/myelinconnect/resting/preprocessed/'+subject+'/'+session+'/'
        out_dir='/scr/ilz3/myelinconnect/final/rest1_1/'
        raw_file=glob(base_dir+'realignment/'+subject+'*'+session+'_roi.nii.gz')[0]
        moco_file=glob(base_dir+'realignment/corr_'+subject+'*'+session+'_roi.nii.gz')[0]
        brain_mask=glob(base_dir+'mask/'+subject+'*T1_Images_mask_fixed_trans.nii.gz')[0]
        wm_mask=base_dir+'mask/wm_mask_trans.nii.gz'
        csf_mask=base_dir+'mask/csf_mask_trans.nii.gz'
        motion_file_12=base_dir+'confounds/motion_regressor_der1_ord1.txt'
        #motion_file_24=base_dir+'confounds/motion_regressor_der1_ord2.txt'
        artefact_file=glob(base_dir+'confounds/art.corr_'+subject+'*'+session+'_roi_outliers.txt')[0]
        
        
        out_file=out_dir+subject+'_'+session+'_denoised.nii.gz'
        confound_file=base_dir+'confounds/all_confounds.txt'
        #compcor_file=base_dir+'confounds/compcor_regressor.txt'

        # reload niftis to round affines so that nilearn doesn't complain
        wm_nii=nb.Nifti1Image(nb.load(wm_mask).get_data(), np.around(nb.load(wm_mask).get_affine(), 2), nb.load(wm_mask).get_header())
        csf_nii=nb.Nifti1Image(nb.load(csf_mask).get_data(),np.around(nb.load(csf_mask).get_affine(), 2), nb.load(csf_mask).get_header())
        moco_nii=nb.Nifti1Image(nb.load(moco_file).get_data(),np.around(nb.load(moco_file).get_affine(), 2), nb.load(moco_file).get_header())

        # infer shape of confound array
        confound_len = nb.load(moco_file).get_data().shape[3]
        
        # create regressors for constant, linear and quadratic trend
#         trend_regressor = np.ones((confound_len, 1))
#         for i in range(2):
#             trend_regressor = np.hstack((trend_regressor, legendre(i + 1)(np.linspace(-1, 1, confound_len))[:, None]))
        
        # create outlier regressors
        outlier_regressor = np.empty((confound_len,1))
        try:
            outlier_val = np.genfromtxt(artefact_file)
        except IOError:
            outlier_val = np.empty((0))
        for index in np.atleast_1d(outlier_val):
            outlier_vector = np.zeros((confound_len, 1))
            outlier_vector[index] = 1
            outlier_regressor = np.hstack((outlier_regressor, outlier_vector))
        
        outlier_regressor = outlier_regressor[:,1::]
        
        # load motion regressors
        motion_regressor_12=np.genfromtxt(motion_file_12)
        #motion_regressor_24=np.genfromtxt(motion_file_24)
        
        # extract high variance confounds in wm/csf masks from motion corrected data
        wm_regressor=high_variance_confounds(moco_nii, mask_img=wm_nii, detrend=True)
        csf_regressor=high_variance_confounds(moco_nii, mask_img=csf_nii, detrend=True)
        
        # extract high variance confounds from unprocessed data
        highvar_regressor=high_variance_confounds(raw_file, detrend=True)
        
        # load compcor regressors
        #compcor_regressor=np.genfromtxt(compcor_file)
        
        # create Nifti Masker for denoising
        denoiser=NiftiMasker(mask_img=brain_mask, standardize=True, detrend=True, high_pass=0.01, low_pass=0.1, t_r=3.0)
        
        # nilearn wmcsf, moco 12
        confounds=np.hstack((outlier_regressor,wm_regressor, csf_regressor,motion_regressor_12))
        denoised_data=denoiser.fit_transform(moco_file, confounds=confounds)
        denoised_img=denoiser.inverse_transform(denoised_data)
        nb.save(denoised_img, out_file)
        np.savetxt((confound_file), confounds, fmt="%.10f")
        

#         # nilearn wmcsf, moco 12, highvar
#         confounds=np.hstack((outlier_regressor,wm_regressor, csf_regressor,motion_regressor_12, highvar_regressor))
#         denoised_data=denoiser.fit_transform(moco_file, confounds=confounds)
#         denoised_img=denoiser.inverse_transform(denoised_data)
#         nb.save(denoised_img, out_file%'nlwmcsf_moco12_highvar')
#         np.savetxt((confound_file%'nlwmcsf_moco12_highvar'), confounds, fmt="%.10f")