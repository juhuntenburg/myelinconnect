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
    inputnode=Node(util.IdentityInterface(fields=['orig_ts',
                                                  'epi_mean',
                                                  'anat_brain',
                                                  'mat_moco',
                                                  'epi2anat_itk',
                                                  'epi2anat_nonlin',
                                                  'resolution']),
                   name='inputnode')
    
    # outputnode                                     
    outputnode=Node(util.IdentityInterface(fields=['trans_ts', 
                                                   'trans_ts_mean',
                                                   'resamp_brain']),
                    name='outputnode')
    
    #resample anatomy
    resample = Node(fsl.FLIRT(datatype='float',
                              out_file='t1_resampled.nii.gz'),
                       name = 'resample_anat')
    transform_ts.connect([(inputnode, resample, [('anat_brain', 'in_file'),
                                                 ('anat_brain', 'reference'),
                                                 ('resolution', 'apply_isoxfm')
                                                 ]),
                          (resample, outputnode, [('out_file', 'resamp_brain')])
                          ])
    
    # split timeseries in single volumes
    split=Node(fsl.Split(dimension='t',
                         out_base_name='timeseries'),
                 name='split')
    
    transform_ts.connect([(inputnode, split, [('orig_ts','in_file')])])
    
    # transform moco to itk
    moco_itk = MapNode(interface=c3.C3dAffineTool(fsl2ras=True,
                                          itk_transform='moco.txt'), 
                       iterfield=['transform_file', 'source_file'],
                       name='moco_itk')
    
    transform_ts.connect([(inputnode, moco_itk, [('mat_moco', 'transform_file'),
                                                 ('epi_mean', 'reference_file')]),
                          (split, moco_itk, [('out_files', 'source_file')])
                          ])               
    
    # make list of transforms
    def makelist(moco, lin, nonlin):
        transformlist=[nonlin[0], lin, moco]
        return transformlist
    
    transformlist = MapNode(interface=util.Function(input_names=['moco', 'lin', 'nonlin'],
                                        output_names=['transformlist'],
                                        function=makelist),
                            iterfield=['moco'], 
                            name='transformlist')
    
    transform_ts.connect([(inputnode, transformlist, [('epi2anat_itk', 'lin'),
                                                      ('epi2anat_nonlin', 'nonlin')]),
                          (moco_itk, transformlist, [('itk_transform', 'moco')])
                          ])
    
    # apply all transforms
    applytransform = MapNode(ants.ApplyTransforms(dimension=3,
                                                  output_image='rest_coregistered.nii.gz',
                                                  interpolation = 'BSpline'),
                             iterfield=['input_image', 'transforms'],
                             name='applytransform')
       
    transform_ts.connect([(split, applytransform, [('out_files', 'input_image')]),
                          (resample, applytransform, [('out_file', 'reference_image')]),
                          (transformlist, applytransform, [('transformlist', 'transforms')])
                          ])
       
    # re-concatenate volumes
    merge=Node(fsl.Merge(dimension='t',
                         merged_file='rest_coregistered.nii.gz'),
               name='merge')
    transform_ts.connect([(applytransform ,merge,[('output_image','in_files')]),
                          (merge, outputnode, [('merged_file', 'trans_ts')])])
    
    
    # calculate new mean
    tmean = Node(fsl.maths.MeanImage(dimension='T',
                                     out_file='rest_coregistered_mean.nii.gz'),
                 name='tmean')

    transform_ts.connect([(merge, tmean, [('merged_file', 'in_file')]),
                          (tmean, outputnode, [('out_file', 'trans_ts_mean')])
                          ])
    
    return transform_ts