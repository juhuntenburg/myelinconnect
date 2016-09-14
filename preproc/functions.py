def fix_hdr(data_file, header_file):
    '''
    Overwrites the header of data_file with the header of header_file
    USE WITH CAUTION
    '''
    
    import nibabel as nb
    import os
    from nipype.utils.filemanip import split_filename
    
    data=nb.load(data_file).get_data()
    hdr=nb.load(header_file).get_header()
    affine=nb.load(header_file).get_affine()
    
    new_file=nb.Nifti1Image(data, affine, hdr)
    _, base, _ = split_filename(data_file)
    nb.save(new_file, base + "_fixed.nii.gz")
    return os.path.abspath(base + "_fixed.nii.gz")


def nilearn_denoise(in_file, brain_mask, wm_mask, csf_mask,
                      motreg_file, outlier_file,
                      bandpass, tr ):
    """Clean time series using Nilearn high_variance_confounds to extract 
    CompCor regressors and NiftiMasker for regression of all nuissance regressors,
    detrending, normalziation and bandpass filtering.
    """
    import numpy as np
    import nibabel as nb
    import os
    from nilearn.image import high_variance_confounds
    from nilearn.input_data import NiftiMasker
    from nipype.utils.filemanip import split_filename

    # reload niftis to round affines so that nilearn doesn't complain
    wm_nii=nb.Nifti1Image(nb.load(wm_mask).get_data(), np.around(nb.load(wm_mask).get_affine(), 2), nb.load(wm_mask).get_header())
    csf_nii=nb.Nifti1Image(nb.load(csf_mask).get_data(), np.around(nb.load(csf_mask).get_affine(), 2), nb.load(csf_mask).get_header())
    time_nii=nb.Nifti1Image(nb.load(in_file).get_data(),np.around(nb.load(in_file).get_affine(), 2), nb.load(in_file).get_header())
        
    # infer shape of confound array
    # not ideal
    confound_len = nb.load(in_file).get_data().shape[3]
    
    # create outlier regressors
    outlier_regressor = np.empty((confound_len,1))
    try:
        outlier_val = np.genfromtxt(outlier_file)
    except IOError:
        outlier_val = np.empty((0))
    for index in np.atleast_1d(outlier_val):
        outlier_vector = np.zeros((confound_len, 1))
        outlier_vector[index] = 1
        outlier_regressor = np.hstack((outlier_regressor, outlier_vector))
    
    outlier_regressor = outlier_regressor[:,1::]
        
    # load motion regressors
    motion_regressor=np.genfromtxt(motreg_file)
    
    # extract high variance confounds in wm/csf masks from motion corrected data
    wm_regressor=high_variance_confounds(time_nii, mask_img=wm_nii, detrend=True)
    csf_regressor=high_variance_confounds(time_nii, mask_img=csf_nii, detrend=True)
    
    # create Nifti Masker for denoising
    denoiser=NiftiMasker(mask_img=brain_mask, standardize=True, detrend=True, high_pass=bandpass[1], low_pass=bandpass[0], t_r=tr)
    
    # denoise and return denoise data to img
    confounds=np.hstack((outlier_regressor,wm_regressor, csf_regressor, motion_regressor))
    denoised_data=denoiser.fit_transform(in_file, confounds=confounds)
    denoised_img=denoiser.inverse_transform(denoised_data)
        
    # save  
    _, base, _ = split_filename(in_file)
    img_fname = base + '_denoised.nii.gz'
    nb.save(denoised_img, img_fname)
    
    confound_fname = os.path.join(os.getcwd(), "all_confounds.txt")
    np.savetxt(confound_fname, confounds, fmt="%.10f")
    
    return os.path.abspath(img_fname), confound_fname


    
'''
======================================
Functions copied from Nipype workflows
======================================
'''

def selectindex(files, idx):
    import numpy as np
    from nipype.utils.filemanip import filename_to_list, list_to_filename
    return list_to_filename(np.array(filename_to_list(files))[idx].tolist())

def median(in_files):
    """Computes an average of the median of each realigned timeseries
    Parameters
    ----------
    in_files: one or more realigned Nifti 4D time series
    Returns
    -------
    out_file: a 3D Nifti file
    """
    import nibabel as nb
    import numpy as np
    import os
    from nipype.utils.filemanip import filename_to_list
    from nipype.utils.filemanip import split_filename
    
    average = None
    for idx, filename in enumerate(filename_to_list(in_files)):
        img = nb.load(filename)
        data = np.median(img.get_data(), axis=3)
        if average is None:
            average = data
        else:
            average = average + data
    median_img = nb.Nifti1Image(average/float(idx + 1),
                                img.get_affine(), img.get_header())
    #filename = os.path.join(os.getcwd(), 'median.nii.gz')
    #median_img.to_filename(filename)
    _, base, _ = split_filename(filename_to_list(in_files)[0])
    nb.save(median_img, base + "_median.nii.gz")
    return os.path.abspath(base + "_median.nii.gz")
    return filename

def get_info(dicom_file):
    """Given a Siemens dicom file return metadata
    Returns
    -------
    RepetitionTime
    Slice Acquisition Times
    Spacing between slices
    """
    from dcmstack.extract import default_extractor
    import numpy as np
    from dicom import read_file
    from nipype.utils.filemanip import filename_to_list
    
    meta = default_extractor(read_file(filename_to_list(dicom_file)[0],
                                       stop_before_pixels=True,
                                       force=True))
    
    TR=meta['RepetitionTime']/1000.
    slice_times_pre=meta['CsaImage.MosaicRefAcqTimes']
    slice_times = (np.array(slice_times_pre)/1000.).tolist()
    slice_thickness = meta['SpacingBetweenSlices']
    
    return TR, slice_times, slice_thickness


def strip_rois_func(in_file, t_min):
    import numpy as np
    import nibabel as nb
    import os
    from nipype.utils.filemanip import split_filename
    nii = nb.load(in_file)
    new_nii = nb.Nifti1Image(nii.get_data()[:,:,:,t_min:], nii.get_affine(), nii.get_header())
    new_nii.set_data_dtype(np.float32)
    _, base, _ = split_filename(in_file)
    nb.save(new_nii, base + "_roi.nii.gz")
    return os.path.abspath(base + "_roi.nii.gz")


def motion_regressors(motion_params, order=0, derivatives=1):
    """Compute motion regressors upto given order and derivative
    motion + d(motion)/dt + d2(motion)/dt2 (linear + quadratic)
    """
    from nipype.utils.filemanip import filename_to_list
    import numpy as np
    import os
    
    out_files = []
    for idx, filename in enumerate(filename_to_list(motion_params)):
        params = np.genfromtxt(filename)
        out_params = params
        for d in range(1, derivatives + 1):
            cparams = np.vstack((np.repeat(params[0, :][None, :], d, axis=0),
                                 params))
            out_params = np.hstack((out_params, np.diff(cparams, d, axis=0)))
        out_params2 = out_params
        for i in range(2, order + 1):
            out_params2 = np.hstack((out_params2, np.power(out_params, i)))
        filename = os.path.join(os.getcwd(), "motion_regressor_der%d_ord%d.txt" % (derivatives, order))
        np.savetxt(filename, out_params2, fmt="%.10f")
        out_files.append(filename)
    return out_files