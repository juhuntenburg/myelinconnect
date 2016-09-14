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
                                                  ]),
                   name='inputnode')
    
    # outputnode                                     
    outputnode=Node(util.IdentityInterface(fields=['uni_lowres',
                                                   'epi2lowres',
                                                   'epi2lowres_mat',
                                                   'epi2lowres_dat',
                                                   'highres2lowres',
                                                   'highres2lowres_mat',
                                                   'highres2lowres_dat',
                                                   'epi2highres_lin',
                                                   'epi2highres_lin_mat',
                                                   'epi2highres_lin_itk'
                                                   ]),
                    name='outputnode')
    
    
    
    # convert mgz head file for reference
    fs_import = Node(interface=nio.FreeSurferSource(),
                     name = 'fs_import')
    
    brain_convert=Node(fs.MRIConvert(out_type='niigz', 
                                     out_file='uni_lowres.nii.gz'),
                       name='brain_convert')
    
    coreg.connect([(inputnode, fs_import, [('fs_subjects_dir','subjects_dir'),
                                            ('fs_subject_id', 'subject_id')]),
                   (fs_import, brain_convert, [('brain', 'in_file')]),
                   (brain_convert, outputnode, [('out_file', 'uni_lowres')])
                   ])
    
    
    # linear registration epi median to lowres mp2rage with bbregister
    bbregister_epi = Node(fs.BBRegister(contrast_type='t2',
                                    out_fsl_file='epi2lowres.mat',
                                    out_reg_file='epi2lowres.dat',
                                    registered_file='epi2lowres.nii.gz',
                                    init='fsl',
                                    epi_mask=True
                                    ),
                    name='bbregister_epi')
    
    coreg.connect([(inputnode, bbregister_epi, [('fs_subjects_dir', 'subjects_dir'),
                                                ('fs_subject_id', 'subject_id'),
                                                ('epi_median', 'source_file')]),
                   (bbregister_epi, outputnode, [('out_fsl_file', 'epi2lowres_mat'),
                                             ('out_reg_file', 'epi2lowres_dat'),
                                             ('registered_file', 'epi2lowres')
                                             ])
                   ])
    

    # linear register highres mp2rage to lowres mp2rage
    bbregister_anat = Node(fs.BBRegister(contrast_type='t1',
                                    out_fsl_file='highres2lowres.mat',
                                    out_reg_file='highres2lowres.dat',
                                    registered_file='highres2lowres.nii.gz',
                                    init='fsl'
                                    ),
                    name='bbregister_anat')

    coreg.connect([(inputnode, bbregister_anat, [('fs_subjects_dir', 'subjects_dir'),
                                                ('fs_subject_id', 'subject_id'),
                                                ('uni_highres', 'source_file')]),
                   (bbregister_anat, outputnode, [('out_fsl_file', 'highres2lowres_mat'),
                                             ('out_reg_file', 'highres2lowres_dat'),
                                             ('registered_file', 'highres2lowres')
                                             ])
                   ])

    # invert highres2lowres transform
    invert = Node(fsl.ConvertXFM(invert_xfm=True),
                  name='invert')
    coreg.connect([(bbregister_anat, invert, [('out_fsl_file', 'in_file')])])
    
    # concatenate epi2highres transforms
    concat = Node(fsl.ConvertXFM(concat_xfm=True,
                                 out_file='epi2highres_lin.mat'),
                  name='concat')
    coreg.connect([(bbregister_epi, concat, [('out_fsl_file', 'in_file')]),
                   (invert, concat, [('out_file', 'in_file2')]),
                   (concat, outputnode, [('out_file', 'epi2higres_lin_mat')])])

    # convert epi2highres transform into itk format
    itk = Node(interface=c3.C3dAffineTool(fsl2ras=True,
                                          itk_transform='epi2highres_lin.txt'), 
                     name='itk')
    
    coreg.connect([(inputnode, itk, [('epi_median', 'source_file'),
                                     ('uni_highres', 'reference_file')]),
                   (concat, itk, [('out_file', 'transform_file')]),
                   (itk, outputnode, [('itk_transform', 'epi2highres_lin_itk')])
                   ])


    # transform epi to highres
    epi2highres = Node(ants.ApplyTransforms(dimension=3,
                                            output_image='epi2highres_lin.nii.gz',
                                            interpolation = 'BSpline',
                                            ),
                       name='epi2highres')
    
    coreg.connect([(inputnode, epi2highres, [('uni_highres', 'reference_image'),
                                             ('epi_median', 'input_image')]),
                   (itk, epi2highres, [('itk_transform', 'transforms')]),
                   (epi2highres, outputnode, [('output_image', 'epi2highres_lin')])])


    return coreg