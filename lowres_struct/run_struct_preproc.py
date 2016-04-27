from nipype.pipeline.engine import Node, Workflow
import nipype.interfaces.io as nio
from struct_preproc.mp2rage import create_mp2rage_pipeline
from struct_preproc.reconall import create_reconall_pipeline
from struct_preproc.mgzconvert import create_mgzconvert_pipeline
#from ants import create_normalize_pipeline
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.c3 as c3


def create_structural(subject, working_dir, data_dir, freesurfer_dir, out_dir):
    
    # main workflow
    struct_preproc = Workflow(name='mp2rage_preproc')
    struct_preproc.base_dir = working_dir
    struct_preproc.config['execution']['crashdump_dir'] = struct_preproc.base_dir + "/crash_files"
    
    # select files
    templates={'inv2': 'raw/mp2rage/inv2.nii.gz',
               't1map': 'raw/mp2rage/t1map.nii.gz',
               'uni': 'raw/mp2rage/uni.nii.gz',
               'uni_highres': 'raw/highres/uni.nii.gz'}
    selectfiles = Node(nio.SelectFiles(templates,
                                       base_directory=data_dir),
                       name="selectfiles")
    
    # workflow for mp2rage background masking
    mp2rage=create_mp2rage_pipeline()
    
    # workflow to run freesurfer reconall
    reconall=create_reconall_pipeline()
    reconall.inputs.inputnode.fs_subjects_dir=freesurfer_dir
    reconall.inputs.inputnode.fs_subject_id=subject
    
    # workflow to get brain, head and wmseg from freesurfer and convert to nifti
    mgzconvert=create_mgzconvert_pipeline()
    
    # workflow to normalize anatomy to standard space
    #normalize=create_normalize_pipeline()
    #normalize.inputs.inputnode.standard = standard_brain
    
    # register highres to lowres
    bbregister_anat = Node(fs.BBRegister(contrast_type='t1',
                                    out_fsl_file='highres2lowres.mat',
                                    out_reg_file='highres2lowres.dat',
                                    registered_file='uni_highres2lowres.nii.gz',
                                    init='fsl'
                                    ),
                    name='bbregister_anat')
    
    # convert transform to itk
    itk_anat = Node(interface=c3.C3dAffineTool(fsl2ras=True,
                                          itk_transform='highres2lowres.txt'), 
                     name='itk_anat')
    
    #sink to store files
    sink = Node(nio.DataSink(base_directory=out_dir,
                             parameterization=False,
                             substitutions=[('outStripped', 'uni_stripped'),
                                            ('outMasked2', 'uni_masked'),
                                            ('outSignal2', 'background_mask'),
                                            ('outOriginal', 'uni_reoriented'),
                                            ('outMask', 'skullstrip_mask'),
                                            ('transform_Warped', 'T1_brain2std')]),
                 name='sink')
    
    
    # connections
    struct_preproc.connect([(selectfiles, mp2rage, [('inv2', 'inputnode.inv2'),
                                                    ('t1map', 'inputnode.t1map'),
                                                    ('uni', 'inputnode.uni')]),
                            (mp2rage, reconall, [('outputnode.uni_masked', 'inputnode.anat')]),
                            (reconall, mgzconvert, [('outputnode.fs_subject_id', 'inputnode.fs_subject_id'),
                                                    ('outputnode.fs_subjects_dir', 'inputnode.fs_subjects_dir')]),
                            #(mgzconvert, normalize, [('outputnode.anat_brain', 'inputnode.anat')]),
                            (reconall, bbregister_anat, [('outputnode.fs_subjects_dir', 'subjects_dir'),
                                                          ('outputnode.fs_subject_id', 'subject_id')]),
                            (selectfiles, bbregister_anat, [('uni_highres', 'source_file')]),
                            (mgzconvert, itk_anat, [('outputnode.anat_brain', 'reference_file')]),
                            (selectfiles, itk_anat,[('uni_highres', 'source_file')]),
                            (bbregister_anat, itk_anat, [('out_fsl_file', 'transform_file')]),
   
                            (mp2rage, sink, [('outputnode.uni_masked', 'preprocessed.mp2rage.@uni_masked'),
                                             ('outputnode.background_mask', 'preprocessed.mp2rage.@background_mask')
                                             ]),
                            (mgzconvert, sink, [('outputnode.anat_head', 'preprocessed.mp2rage.@head'),
                                                ('outputnode.anat_brain', 'preprocessed.mp2rage.@brain'),
                                                ('outputnode.brain_mask', 'preprocessed.mp2rage.@brain_mask'),
                                                ('outputnode.wmedge', 'preprocessed.mp2rage.@wmedge'),
                                                #('outputnode.wmseg', 'preprocessed.mp2rage.brain_extraction.@wmseg')
                                                ]),                
                            (bbregister_anat, sink, [('out_fsl_file', 'preprocessed.mp2rage.@highres2lowres_mat'),
                                                     ('out_reg_file', 'preprocessed.mp2rage.@highres2lowres_dat'),
                                                     ('registered_file', 'preprocessed.mp2rage.@highres2lowres')]),
                            (itk_anat, sink, [('itk_transform', 'preprocessed.mp2rage.@highres2lowres_itk')])
                            
                            #(normalize, sink, [('outputnode.anat2std', 'preprocessed.mp2rage.normalization.@anat2std'),
                            #                   ('outputnode.anat2std_transforms', 'preprocessed.mp2rage.normalization.@anat2std_transforms'),
                            #                   ('outputnode.std2anat_transforms', 'preprocessed.mp2rage.normalization.@std2anat_transforms')])
                            ])
    #struct_preproc.write_graph(dotfilename='struct_preproc.dot', graph2use='colored', format='pdf', simple_form=True)
    return struct_preproc
    
    
'''
===========================================
''' 
subject=raw_input('subject: ')

working_dir = '/scr/ilz2/7tresting/working_dir/'+subject+'/' 
data_dir = '/scr/ilz2/7tresting/'+subject+'/'
out_dir = '/scr/ilz2/7tresting/'+subject+'/'
freesurfer_dir = '/scr/ilz2/7tresting/freesurfer/' 

struct=create_structural(subject, working_dir, data_dir, freesurfer_dir, out_dir)
struct.run(plugin='MultiProc')
