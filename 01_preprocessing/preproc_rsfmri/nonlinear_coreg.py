from nipype.pipeline.engine import Node, Workflow
import nipype.interfaces.utility as util
import nipype.interfaces.ants as ants
import nipype.interfaces.fsl as fsl
import nipype.interfaces.freesurfer as fs


def create_nonlinear_pipeline(name='nonlinear'):
    
    # workflow
    nonlinear=Workflow(name='nonlinear')
    
    # inputnode
    inputnode=Node(util.IdentityInterface(fields=['t1_highres',
                                                  'epi2highres_lin',
                                                  'epi2highres_lin_itk',
                                                  'fov_mask',
                                                  'brain_mask',
                                                  #'highres2lowres_itk'
                                                  ]),
                   name='inputnode')
    
    # outputnode                                 
    outputnode=Node(util.IdentityInterface(fields=['epi2highres_warp',
                                                   'epi2highres_invwarp',
                                                   'epi2highres_nonlin',
                                                   ]),
                    name='outputnode')
    
#     
#     brainmask = Node(ants.ApplyTransforms(dimension=3,
#                                           invert_transform_flags=[True],
#                                           interpolation = 'NearestNeighbor'),
#                      name='brainmask')
#     
    dil_brainmask = Node(fs.Binarize(min=0.5,
                                 out_type = 'nii.gz',
                                 dilate=15),
                      name='dil_brainmask')

    
    mask_epi = Node(fsl.ApplyMask(out_file='epi2highres_lin_masked.nii.gz'),
                    name='mask_epi')
    
    nonlinear.connect([#(inputnode, brainmask, [('brain_mask', 'input_image'),
                       #                        ('t1_highres', 'reference_image'),
                       #                        ('highres2lowres_itk', 'transforms')]),
                       #(brainmask, dil_brainmask, [('output_image', 'in_file')]),
                       (inputnode, dil_brainmask, [('brain_mask', 'in_file')]),
                       (dil_brainmask, mask_epi, [('binary_file', 'mask_file')]),
                       (inputnode, mask_epi, [('epi2highres_lin', 'in_file')])
                       ])
    
    # transform fov mask and apply to t1       
    transform_fov = Node(ants.ApplyTransforms(dimension=3,
                                              #invert_transform_flags=[True, False],
                                              output_image='fov_mask_highres.nii.gz',
                                              interpolation = 'NearestNeighbor'),
                          'transform_fov')
    
    dilate_fov = Node(fs.Binarize(min=0.5,
                                  dilate=5,
                                  binary_file='fov_mask_highres_dil.nii.gz'),
                      name='dilate_fov')   
    
    
    mask_t1 = Node(fsl.ApplyMask(out_file='t1_fov_masked.nii.gz'),
                    name='mask_t1')
    
    nonlinear.connect([(inputnode, transform_fov, [('fov_mask', 'input_image'),
                                                   ('t1_highres', 'reference_image'),
                                                   ('epi2highres_lin_itk', 'transforms')]),
                       (transform_fov, dilate_fov, [('output_image', 'in_file')]),
                       (dilate_fov, mask_t1, [('binary_file', 'mask_file')]),
                       (inputnode, mask_t1, [('t1_highres', 'in_file')]),
                       ])
    
    
    # normalization with ants
    antsreg = Node(interface = ants.registration.Registration(dimension = 3,
                                                           metric = ['CC'],
                                                           metric_weight = [1.0],
                                                           radius_or_number_of_bins = [4],
                                                           sampling_strategy = ['None'],
                                                           transforms = ['SyN'],
                                                           args = '-g 0.1x1x0.1',
                                                           transform_parameters = [(0.10,3,0)],
                                                           number_of_iterations = [[50,20,10]],
                                                           convergence_threshold = [1e-06],
                                                           convergence_window_size = [10],
                                                           shrink_factors = [[4,2,1]],
                                                           smoothing_sigmas = [[2,1,0]],
                                                           sigma_units = ['vox'],
                                                           use_estimate_learning_rate_once = [True],
                                                           use_histogram_matching = [True],
                                                           collapse_output_transforms=True,
                                                           output_inverse_warped_image = True,
                                                           output_warped_image = True,
                                                           interpolation = 'BSpline'),
                      name = 'antsreg')
    antsreg.plugin_args={'submit_specs': 'request_memory = 20000'}
       
    nonlinear.connect([(mask_epi, antsreg, [('out_file', 'moving_image')]),
                       (mask_t1, antsreg, [('out_file', 'fixed_image')]),
                       (antsreg, outputnode, [('reverse_transforms', 'epi2highres_invwarp'),
                                              ('forward_transforms', 'epi2highres_warp'),
                                              ('warped_image', 'epi2highres_nonlin')])
                        ])
     
    return nonlinear

# test_nonlinear=create_nonlinear_pipeline('nonlinear')
# test_nonlinear.base_dir='/scr/kansas1/huntenburg/7tresting/working/highres_bias_bspline_maskepi/'
# test_nonlinear.config['execution']['crashdump_dir'] = test_nonlinear.base_dir + "/crash_files"
# test_nonlinear.config['execution']['remove_unnecessary_outputs'] = False
# test_nonlinear.inputs.inputnode.anat='/scr/kansas1/huntenburg/7tresting/sub021/preprocessed/coregister/t1_resampled.nii.gz'
# test_nonlinear.inputs.inputnode.epi= '/scr/kansas1/huntenburg/7tresting/sub021/preprocessed/coregister/rest_coregistered_mean.nii.gz'
# #test_nonlinear.inputs.inputnode.anat='/scr/kansas1/huntenburg/7tresting/sub021/highres/t1.nii.gz'
# #test_nonlinear.inputs.inputnode.epi= '/scr/kansas1/huntenburg/7tresting/sub021/coreg_testing/flirt_epi2highres_t1_cr_bbr_onestep.nii.gz'
# test_nonlinear.run()#plugin='CondorDAGMan')