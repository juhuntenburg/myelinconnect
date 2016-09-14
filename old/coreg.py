from nipype.pipeline.engine import Node, Workflow
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
import nipype.interfaces.c3 as c3
import os

def create_coreg_pipeline(name='coreg'):
    
    # fsl output type
    fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
    
    # initiate workflow
    coreg = Workflow(name='coreg')
    
    #inputnode 
    inputnode=Node(util.IdentityInterface(fields=['epi_mean',
                                                  'anat_brain',
                                                  'wmseg'
                                                  ]),
                   name='inputnode')
    
    # outputnode                                     
    outputnode=Node(util.IdentityInterface(fields=['coregistered_mean',
                                                   'epi2anat_mat',
                                                   'epi2anat_itk'
                                                   ]),
                    name='outputnode')
    
    
    #swap dimensions to come closer to T1
    swapdim = Node(fsl.SwapDimensions(new_dims=('-x', '-y', 'z')),
                   name='swapdim')
    
    coreg.connect([(inputnode, swapdim, [('epi_mean', 'in_file')])])
    
    # linear registration with flirt
    epi2anat1 = Node(fsl.FLIRT(dof=6),
           name='epi2anat1')
  
    coreg.connect([(swapdim, epi2anat1, [('out_file', 'in_file')]),
                        (inputnode, epi2anat1, [('anat_brain','reference')])
                        ])
    
    # refine with bbr option
    epi2anat2 = Node(fsl.FLIRT(dof=6,
                               interp='spline',
                               cost='bbr',
                               schedule=os.path.abspath('/usr/share/fsl/5.0/etc/flirtsch/bbr.sch'),
                               out_matrix_file='epi2anat.mat',
                               out_file='rest_lin_coreg.nii.gz'),
           name='epi2anat2')
 
    coreg.connect([(swapdim, epi2anat2, [('out_file', 'in_file')]),
                        (inputnode, epi2anat2, [('anat_brain','reference')]),
                        (epi2anat1, epi2anat2, [('out_matrix_file', 'in_matrix_file')]),
                        (inputnode, epi2anat2, [('wmseg', 'wm_seg')]),
                        (epi2anat2, outputnode, [('out_file', 'coregistered_mean'),
                                                 ('out_matrix_file', 'epi2anat_mat')])
                        ])
    
    # convert transform to itk
    itk = Node(interface=c3.C3dAffineTool(fsl2ras=True,
                                          itk_transform='epi2anat.txt'), 
                     name='itk')
    
    coreg.connect([(inputnode, itk, [('anat_brain', 'reference_file'),
                                     ('epi_mean', 'source_file')]),
                   (epi2anat2, itk, [('out_matrix_file', 'transform_file')]),
                   (itk, outputnode, [('itk_transform', 'epi2anat_itk')])
                   ])

    
    return coreg