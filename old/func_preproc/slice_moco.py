from nipype.pipeline.engine import Node, Workflow
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
import nipype.interfaces.nipy as nipy
import nipype.algorithms.misc as misc

def create_slicemoco_pipeline(name='slicetime_motion_correction'):
    
    # initiate workflow
    slicemoco=Workflow(name='slicetime_motion_correction')
    
    # set fsl output
    fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
    
    # inputnode
    inputnode = Node(util.IdentityInterface(fields=['epi',
                                                    'dicom']),
                     name='inputnode')
    
    # outputnode
    outputnode = Node(util.IdentityInterface(fields=['epi_moco', 
                                                     'par_moco',
                                                     'detrended_file', 
                                                     'epi_median',
                                                     'fov_mask',
                                                     'tr']),
                      name='outputnode')
    
    
    
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
        filename = os.path.join(os.getcwd(), 'median.nii.gz')
        median_img.to_filename(filename)
        return filename

    
    getinfo = Node(util.Function(input_names=['dicom_file'],
                                 output_names=['TR', 'slice_times', 'slice_thickness'],
                                 function=get_info),
                   name='getinfo')
    
    
    
    # simultaneous slice time and motion correction
    realign = Node(nipy.SpaceTimeRealigner(), 
                   name="spacetime_realign")
    realign.inputs.slice_info = 2
    
    
    # detrend timeseries
    tsnr = Node(misc.TSNR(regress_poly=2),
                name='tsnr')
    
    
    median = Node(util.Function(input_names=['in_files'],
                           output_names=['median_file'],
                           function=median),
                  name='median')
    
    # make fov mask
    fov = Node(fsl.maths.MathsCommand(args='-bin',
                                      out_file='fov_mask_lowres.nii.gz'),
               name='fov_mask')
    
    # create connections
    slicemoco.connect([(inputnode, getinfo, [('dicom', 'dicom_file')]),
                       (getinfo, realign, [('slice_times', 'slice_times'),
                                           ('TR', 'tr')]),
                       (getinfo, outputnode, [('TR', 'tr')]),
                       (inputnode, realign, [('epi', 'in_file')]),
                       (realign, outputnode, [('out_file','epi_moco'),
                                              ('par_file','par_moco')]),
                       (realign, tsnr, [('out_file', 'in_file')]),
                       #(tsnr, outputnode, [('detrended_file', 'detrended_file' )]),
                       (tsnr, median, [('detrended_file','in_files')]),
                       (median, outputnode, [('median_file', 'epi_median')]),
                       (median, fov, [('median_file', 'in_file')]),
                       (fov, outputnode, [('out_file', 'fov_mask')])
                  ])
        
        
    return slicemoco

    
    
    