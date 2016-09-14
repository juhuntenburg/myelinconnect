from nipype.pipeline.engine import Node, Workflow
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
import nipype.interfaces.c3 as c3
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.ants as ants
import nipype.interfaces.io as nio
import os

def create_coreg_pipeline(name='coreg'):
    
    # fsl output type
    fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
    
    # initiate workflow
    coreg = Workflow(name='coreg')
    
    #inputnode 
    inputnode=Node(util.IdentityInterface(fields=['epi_median',
                                                  'fs_subjects_dir',
                                                  'fs_subject_id',
                                                  'uni_highres',
                                                  'uni_lowres'
                                                  ]),
                   name='inputnode')
    
    # outputnode                                     
    outputnode=Node(util.IdentityInterface(fields=['uni_lowres',
                                                   'epi2lowres',
                                                   'epi2lowres_mat',
                                                   'epi2lowres_dat',
                                                   'epi2lowres_itk',
                                                   'highres2lowres',
                                                   'highres2lowres_mat',
                                                   'highres2lowres_dat',
                                                   'highres2lowres_itk',
                                                   'epi2highres',
                                                   'epi2highres_itk'
                                                   ]),
                    name='outputnode')
    
    
    
    # convert mgz head file for reference
    fs_import = Node(interface=nio.FreeSurferSource(),
                     name = 'fs_import')
    
    brain_convert=Node(fs.MRIConvert(out_type='niigz'),
                       name='brain_convert')
    
    coreg.connect([(inputnode, fs_import, [('fs_subjects_dir','subjects_dir'),
                                            ('fs_subject_id', 'subject_id')]),
                   (fs_import, brain_convert, [('brain', 'in_file')]),
                   (brain_convert, outputnode, [('out_file', 'uni_lowres')])
                   ])
    
    
    # biasfield correctio of median epi
    biasfield = Node(interface = ants.segmentation.N4BiasFieldCorrection(save_bias=True),
                     name='biasfield')
    
    coreg.connect([(inputnode, biasfield, [('epi_median', 'input_image')])])
    
    
    # linear registration epi median to lowres mp2rage with bbregister
    bbregister_epi = Node(fs.BBRegister(contrast_type='t2',
                                    out_fsl_file='epi2lowres.mat',
                                    out_reg_file='epi2lowres.dat',
                                    #registered_file='epi2lowres.nii.gz',
                                    init='fsl',
                                    epi_mask=True
                                    ),
                    name='bbregister_epi')
    
    coreg.connect([(inputnode, bbregister_epi, [('fs_subjects_dir', 'subjects_dir'),
                                            ('fs_subject_id', 'subject_id')]),
                   (biasfield, bbregister_epi, [('output_image', 'source_file')]),
                   (bbregister_epi, outputnode, [('out_fsl_file', 'epi2lowres_mat'),
                                             ('out_reg_file', 'epi2lowres_dat'),
                                             ('registered_file', 'epi2lowres')
                                             ])
                   ])
    
    
    # convert transform to itk
    itk_epi = Node(interface=c3.C3dAffineTool(fsl2ras=True,
                                          itk_transform='epi2lowres.txt'), 
                     name='itk')
     
    coreg.connect([(brain_convert, itk_epi, [('out_file', 'reference_file')]),
                   (biasfield, itk_epi, [('output_image', 'source_file')]),
                   (bbregister_epi, itk_epi, [('out_fsl_file', 'transform_file')]),
                   (itk_epi, outputnode, [('itk_transform', 'epi2lowres_itk')])
                   ])
    
    
    
    # linear register highres highres mp2rage to lowres mp2rage
    bbregister_anat = Node(fs.BBRegister(contrast_type='t1',
                                    out_fsl_file='highres2lowres.mat',
                                    out_reg_file='highres2lowres.dat',
                                    #registered_file='uni_highres2lowres.nii.gz',
                                    init='fsl'
                                    ),
                    name='bbregister_anat')

    coreg.connect([(inputnode, bbregister_epi, [('fs_subjects_dir', 'subjects_dir'),
                                                ('fs_subject_id', 'subject_id'),
                                                ('uni_highres', 'source_file')]),
                   (bbregister_epi, outputnode, [('out_fsl_file', 'highres2lowres_mat'),
                                             ('out_reg_file', 'highres2lowres_dat'),
                                             ('registered_file', 'highres2lowres')
                                             ])
                   ])
    
    
    # convert transform to itk
    itk_anat = Node(interface=c3.C3dAffineTool(fsl2ras=True,
                                          itk_transform='highres2lowres.txt'), 
                     name='itk_anat')
    
    coreg.connect([(inputnode, itk_anat, [('uni_highres', 'source_file')]),
                   (brain_convert, itk_anat, [('out_file', 'reference_file')]),
                   (bbregister_anat, itk_epi, [('out_fsl_file', 'transform_file')]),
                   (itk_anat, outputnode, [('itk_transform', 'highres2lowres_itk')])
                   ])

    
    # merge transforms into list
    translist = Node(util.Merge(2),
                     name='translist')
    coreg.connect([(itk_anat, translist, [('highres2lowres_itk', 'in1')]),
                   (itk_epi, translist, [('itk_transform', 'in2')]),
                   (translist, outputnode, [('out', 'epi2highres_itk')])])



    # transform epi to highres
    epi2highres = Node(ants.ApplyTransforms(dimension=3,
                                            #output_image='epi2highres.nii.gz',
                                            interpolation = 'BSpline',
                                            invert_transform_flags=[True, False]),
                       name='epi2highres')
    
    coreg.connect([(inputnode, epi2highres, [('uni_highres', 'reference_image')]),
                   (biasfield, epi2highres, [('output_image', 'input_image')]),
                   (translist, epi2highres, [('out', 'transforms')]),
                   (epi2highres, outputnode, [('output_image', 'epi2highres')])])


    
    
    return coreg