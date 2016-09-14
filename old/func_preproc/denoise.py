from nipype.pipeline.engine import Node, Workflow, MapNode
import nipype.interfaces.fsl as fsl
import nipype.interfaces.utility as util
import nipype.algorithms.rapidart as ra
from noise.motreg import motion_regressors
from noise.motionfilter import build_filter1
from noise.compcor import extract_noise_components
#from bandpass import bandpass_filter
import nipype.interfaces.ants as ants
from normalize_timeseries import time_normalizer
from nipype.utils.filemanip import list_to_filename



def create_denoise_pipeline(name='denoise'):

    # workflow
    denoise = Workflow(name='denoise')

    # Define nodes
    inputnode = Node(interface=util.IdentityInterface(fields=['t1_highres_resamp',
                                                              'uni_lowres_brain',
                                                              'uni_lowres_brainmask',
                                                              'highres2lowres_itk',
                                                              'epi_coreg',
                                                              'moco_par',
                                                              'highpass_sigma',
                                                              'lowpass_sigma',
                                                              'tr']),
                        name='inputnode')
    
    outputnode = Node(interface=util.IdentityInterface(fields=['wmcsf_mask',
                                                               'brain_mask_resamp',
                                                               'combined_motion',
                                                               'outlier_files',
                                                               'intensity_files',
                                                               'outlier_stats',
                                                               'outlier_plots',
                                                               'mc_regressor',
                                                               'mc_F',
                                                               'mc_pF',
                                                               'comp_regressor',
                                                               'comp_F',
                                                               'comp_pF',
                                                               'bandpassed_file',
                                                               'normalized_file']),
                     name='outputnode')
    
    
    # run fast to get tissue probability classes
    fast = Node(fsl.FAST(), name='fast')
    denoise.connect([(inputnode, fast, [('uni_lowres_brain', 'in_files')])])
    
    # function to select tissue classes    
    def selectindex(files, idx):
        import numpy as np
        from nipype.utils.filemanip import filename_to_list, list_to_filename
        return list_to_filename(np.array(filename_to_list(files))[idx].tolist())
    
    
    def selectsingle(files, idx):
        return files[idx]
    
   
    # transfrom tissue masks to highres space
    transtissue = MapNode(ants.ApplyTransforms(dimension=3,
                                              invert_transform_flags=[True],
                                              interpolation = 'NearestNeighbor'),
                             iterfield=['input_image'],
                             name='transtissue')
    
    denoise.connect([(fast, transtissue, [(('partial_volume_files', selectindex, [0, 2]),'input_image')]),
                     (inputnode, transtissue, [('highres2lowres_itk', 'transforms'),
                                               ('t1_highres_resamp', 'reference_image')])])
   
    
    # binarize tissue classes
    binarize_tissue = MapNode(fsl.ImageMaths(op_string='-nan -thr 0.99 -ero -bin'),
                        iterfield=['in_file'],
                        name='binarize_tissue')
    
    denoise.connect([(transtissue, binarize_tissue, [('output_image', 'in_file')])])
    
    
    # combine tissue classes to noise mask
    wmcsf_mask = Node(fsl.BinaryMaths(operation='add',
                                        out_file='wmcsf_mask.nii.gz'),
                      name='wmcsf_mask')
    
    denoise.connect([(binarize_tissue, wmcsf_mask, [(('out_file', selectsingle, 0), 'in_file'),
                                                    (('out_file', selectsingle, 1), 'operand_file')]),
                     (wmcsf_mask, outputnode, [('out_file', 'wmcsf_mask')])])
    
    # transfrom brain mask to highres space
    transbrainmask = Node(ants.ApplyTransforms(dimension=3,
                                              invert_transform_flags=[True],
                                              output_image='brainmask2highres.nii.gz',
                                              interpolation = 'NearestNeighbor'),
                          'transbrainmask')
    
    denoise.connect([(inputnode, transbrainmask, [('uni_lowres_brainmask', 'input_image'),
                                                  ('t1_highres_resamp', 'reference_image'),
                                                  ('highres2lowres_itk', 'transforms')]),
                     (transbrainmask, outputnode, [('output_image', 'brain_mask_resamp')])])
#     # binarize resampled brain for masking
#     binarize_brain = Node(fs.Binarize(min=0.5,
#                                 out_type='nii.gz',
#                                 binary_file='brain_mask_resamp.nii.gz'),
#                     name='binarize')
#      
#     denoise.connect([(inputnode, binarize_brain, [('t1_highres_resamp', 'in_file')]),
#                      (binarize_brain, outputnode, [('binary_file', 'brain_mask_resamp')])
#                      ])
     
    
    # mask functional data
#     mask_epi = Node(fsl.ApplyMask(out_file='rest_masked.nii.gz'),
#                     name='mask_epi')
#      
#     denoise.connect([(inputnode, mask_epi, [('epi_moco', 'in_file')]),
#                      (binarize_brain, mask_epi, [('binary_file', 'mask_file')])
#                      ])
    
        
    # perform artefact detection
    artefact=Node(ra.ArtifactDetect(save_plot=True,
                                    use_norm=True,
                                    parameter_source='NiPy',
                                    mask_type='file',
                                    norm_threshold=1,
                                    zintensity_threshold=3,
                                    use_differences=[True,False]
                                    ),
                 name='artefact')
    
    denoise.connect([(inputnode, artefact, [('epi_coreg', 'realigned_files'),
                                            ('moco_par', 'realignment_parameters')]),
                     (transbrainmask, artefact, [('output_image', 'mask_file')]),
                     #(binarize_brain, artefact, [('binary_file', 'mask_file')]),
                     (artefact, outputnode, [('norm_files', 'combined_motion'),
                                             ('outlier_files', 'outlier_files'),
                                             ('intensity_files', 'intensity_files'),
                                             ('statistic_files', 'outlier_stats'),
                                             ('plot_files', 'outlier_plots')])])
    
        
    # Compute motion regressors
    motreg = Node(util.Function(input_names=['motion_params', 'order','derivatives'],
                                output_names=['out_files'],
                                function=motion_regressors),
                  name='getmotionregress')
    denoise.connect([(inputnode, motreg, [('moco_par','motion_params')])])
    
    # Create a filter to remove motion and art confounds
    createfilter1 = Node(util.Function(input_names=['motion_params', 'comp_norm',
                                                    'outliers', 'detrend_poly'],
                                       output_names=['out_files'],
                                       function=build_filter1),
                         name='makemotionbasedfilter')
    createfilter1.inputs.detrend_poly = 2
    denoise.connect([(motreg, createfilter1, [('out_files','motion_params')]),
                     (artefact, createfilter1, [('norm_files', 'comp_norm'),
                                                ('outlier_files', 'outliers')]),
                     (createfilter1, outputnode, [('out_files', 'mc_regressor')])
                     ])
    
    
    # regress out motion and art confounds
    filter1 = Node(fsl.GLM(out_f_name='F_mcart.nii.gz',
                               out_pf_name='pF_mcart.nii.gz',
                               out_res_name='rest_mc_denoised.nii.gz',
                               demean=True),
                   name='filtermotion')
     
    denoise.connect([(inputnode, filter1, [('epi_coreg', 'in_file')]),
                    (createfilter1, filter1, [(('out_files', list_to_filename), 'design')]),
                    (filter1, outputnode, [('out_f', 'mc_F'),
                                           ('out_pf', 'mc_pF')])])
    

    
    # create filter with compcor components
    createfilter2 = Node(util.Function(input_names=['realigned_file', 'mask_file',
                                                    'num_components',
                                                    'extra_regressors'],
                                       output_names=['out_files'],
                                       function=extract_noise_components),
                         name='makecompcorfilter')
    createfilter2.inputs.num_components = 6
    denoise.connect([#(createfilter1, createfilter2, [(('out_files', list_to_filename),'extra_regressors')]),
                     (filter1, createfilter2, [('out_res', 'realigned_file')]),
                     (wmcsf_mask, createfilter2, [('out_file','mask_file')]),
                     (createfilter2, outputnode, [('out_files','comp_regressor')]),
                     ])


    
    # regress compcorr and other noise components
    filter2 = Node(fsl.GLM(out_f_name='F_noise.nii.gz',
                           out_pf_name='pF_noise.nii.gz',
                           out_res_name='rest_denoised.nii.gz',
                           demean=True),
                   name='filternoise')
    denoise.connect([(filter1, filter2, [('out_res', 'in_file')]),
                     (createfilter2, filter2, [('out_files', 'design')]),
                     (transbrainmask, filter2, [('output_image', 'mask')]),
                     (filter2, outputnode, [('out_f', 'comp_F'),
                                            ('out_pf', 'comp_pF')])
                     ])
    
#     # function to build reciproc
#     def recip(tr):
#         fs=1./tr
#         return fs
#     
#     # bandpass denoised data
#     bandpass = Node(util.Function(input_names=['files', 'lowpass_freq',
#                                                'highpass_freq', 'fs'],
#                                    output_names=['out_files'],
#                                    function=bandpass_filter),
#                     name='bandpass_unsmooth')
#     
#     denoise.connect([(inputnode, bandpass, [(('TR',recip), 'fs'),
#                                             ('highpass', 'highpass_freq'),
#                                             ('lowpass', 'lowpass_freq')]),
#                      (filter2, bandpass, [('out_res', 'files')]),
#                      (bandpass, outputnode, [('out_files', 'bandpassed_file')])
#                     ])
#                     
# 
#     denoise.connect
    
    
    # bandpass filter denoised file
    bandpass_filter = Node(fsl.TemporalFilter(out_file='rest_denoised_bp.nii.gz'),
                              name='bandpass_filter')

    denoise.connect([(inputnode, bandpass_filter,[( 'highpass_sigma','highpass_sigma'),
                                                  ('lowpass_sigma', 'lowpass_sigma')]),
                     (filter2, bandpass_filter, [('out_res', 'in_file')])
                     ])
    
    
    # normalize scans
    normalize_time=Node(util.Function(input_names=['in_file','tr'],
                                         output_names=['out_file'],
                                         function=time_normalizer),
                           name='normalize_time')
    
    
    denoise.connect([(inputnode, normalize_time, [('tr', 'tr')]),
                     (bandpass_filter, normalize_time, [('out_file', 'in_file')]),
                     (normalize_time, outputnode, [('out_file', 'normalized_file')])
                     ])
    
    
    return denoise
