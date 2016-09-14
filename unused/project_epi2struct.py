import pandas as pd
from nipype.pipeline.engine import Node, Workflow, MapNode
import nipype.interfaces.utility as util
import nipype.interfaces.io as nio
import nipype.interfaces.fsl as fsl
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.afni as afni
import nipype.interfaces.nipy as nipy
import nipype.algorithms.rapidart as ra
from nipype.algorithms.misc import TSNR
import nipype.interfaces.ants as ants
import nilearn.image as nli
from functions import strip_rois_func, get_info, median, motion_regressors, extract_noise_components, selectindex, fix_hdr
from linear_coreg import create_coreg_pipeline
from nonlinear_coreg import create_nonlinear_pipeline


# read in subjects and file names
df=pd.read_csv('/scr/ilz3/myelinconnect/subjects.csv', header=0)
subjects_db=list(df['DB'])

# sessions to loop over
sessions=['rest1_1'] # ,'rest1_2', 'rest2_1', 'rest2_2']

# directories
working_dir = '/scr/ilz3/myelinconnect/working_dir/' 
data_dir= '/scr/ilz3/myelinconnect/'
final_dir = '/scr/ilz3/myelinconnect/final_struct_space/'

# set fsl output type to nii.gz
fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

# main workflow
epi2struct = Workflow(name='epi2struct')
epi2struct.base_dir = working_dir
epi2struct.config['execution']['crashdump_dir'] = epi2struct.base_dir + "/crash_files"

# iterate over subjects
subject_infosource = Node(util.IdentityInterface(fields=['subject']), 
                  name='subject_infosource')
subject_infosource.iterables=[('subject', subjects_db)]

# iterate over sessions
session_infosource = Node(util.IdentityInterface(fields=['session']), 
                  name='session_infosource')
session_infosource.iterables=[('session', sessions)]

# select files
templates={'rest': 'final/rest1_1/rest_denoised/{subject}_{session}_denoised.nii.gz',
           'epi2highres_lin_itk' : 'resting/preprocessed/{subject}/{session}/registration/epi2highres_lin.txt',
           'epi2highres_warp':'resting/preprocessed/{subject}/{session}/registration/transform0Warp.nii.gz',
           't1_highres' : 'struct/t1/{subject}*T1_Images_merged.nii.gz'
           }    
selectfiles = Node(nio.SelectFiles(templates, base_directory=data_dir),
                   name="selectfiles")

epi2struct.connect([(subject_infosource, selectfiles, [('subject', 'subject')]),
                 (session_infosource, selectfiles, [('session', 'session')])
                 ])


# resample t1_highres to func resolution
resamp=Node(fsl.FLIRT(apply_isoxfm=1.5,
                   interp='sinc'),
         name='resamp')

epi2struct.connect([(selectfiles, resamp, [('t1_highres', 'in_file'),
                                           ('t1_highres', 'reference')])])

# merge func2struct transforms into list
translist_forw = Node(util.Merge(2),name='translist_forw')
epi2struct.connect([(selectfiles, translist_forw, [('epi2highres_lin_itk', 'in2')]),
                 (selectfiles, translist_forw, [('epi2highres_warp', 'in1')])])
   

# project structural files to functional space
func2struct = Node(ants.ApplyTransforms(invert_transform_flags=[False, False],
                                        input_image_type=3,
                                        dimension=3,
                                        interpolation='CosineWindowedSinc'),
                    name='func2struct')
    
epi2struct.connect([(selectfiles, func2struct, [('rest', 'input_image')]),
                 (translist_forw, func2struct, [('out', 'transforms')]),
                 (resamp, func2struct, [('out_file', 'reference_image')]),
                 ])

  
# sink relevant files
final_sink = Node(nio.DataSink(parameterization=False,
                               base_directory=final_dir),
             name='final_sink')

epi2struct.connect([(session_infosource, final_sink, [('session', 'container')]),
                   (func2struct, final_sink, [('output_image', '@func')])
                   ])
    
epi2struct.run(plugin='MultiProc', plugin_args={'n_procs' : 9})
