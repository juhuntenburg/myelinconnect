from nipype.pipeline.engine import MapNode, Node, Workflow
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
import nipype.interfaces.c3 as c3
import nipype.interfaces.ants as ants


def create_transform_pipeline(name='transfrom_timeseries'):
    
    # set fsl output type
    fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
    
    # initiate workflow
    transform_ts = Workflow(name='transform_timeseries')
    
    # inputnode
    inputnode=Node(util.IdentityInterface(fields=['epi_moco',
                                                  't1_highres',
                                                  'epi2highres_itk',
                                                  'epi2highres_nonlin',
                                                  'resolution']),
                   name='inputnode')
    
    # outputnode                                     
    outputnode=Node(util.IdentityInterface(fields=['trans_ts', 
                                                   'trans_ts_mean',
                                                   't1_highres_resamp']),
                    name='outputnode')
    
    #resample anatomy
    resample = Node(fsl.FLIRT(datatype='float',
                              out_file='t1_highres_resamp.nii.gz'),
                       name = 'resample_anat')
    transform_ts.connect([(inputnode, resample, [('t1_highres', 'in_file'),
                                                 ('t1_highres', 'reference'),
                                                 ('resolution', 'apply_isoxfm')
                                                 ]),
                          (resample, outputnode, [('out_file', 't1_highres_resamp')])
                          ])
    
    # merge transforms into list
    translist = Node(util.Merge(2),
                     name='translist')
    transform_ts.connect([(inputnode, translist, [('epi2highres_itk', 'in2'),
                                                  ('epi2highres_nonlin', 'in1')])])
    
    
    # apply all transforms
    applytransform = Node(ants.ApplyTransforms(input_image_type = 3,
                                               output_image='rest_coregistered.nii.gz',
                                               interpolation = 'BSpline',
                                               invert_transform_flags=[False, True, False]),
                          name='applytransform')
       
    transform_ts.connect([(inputnode, applytransform, [('epi_moco', 'input_image')]),
                          (resample, applytransform, [('out_file', 'reference_image')]),
                          (translist, applytransform, [('out', 'transforms')]),
                          (applytransform, outputnode, [('output_image', 'trans_ts')])
                          ])
       
    # calculate new mean
    tmean = Node(fsl.maths.MeanImage(dimension='T',
                                     out_file='rest_coregistered_mean.nii.gz'),
                 name='tmean')

    transform_ts.connect([(applytransform, tmean, [('output_image', 'in_file')]),
                          (tmean, outputnode, [('out_file', 'trans_ts_mean')])
                          ])
    
    return transform_ts