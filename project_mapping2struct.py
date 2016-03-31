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
subjects = pd.read_csv('/scr/ilz3/myelinconnect/subjects.csv')
subjects=list(subjects['DB'])
subjects.remove('KSMT')

sessions = ['rest1_1', 'rest1_2', 'rest2_1', 'rest2_2']

# directories
working_dir = '/scr/ilz3/myelinconnect/working_dir/' 
data_dir= '/scr/ilz3/myelinconnect/'
final_dir = '/scr/ilz3/myelinconnect/mappings/rest2highres/'

# set fsl output type to nii.gz
fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

# main workflow
mapping2struct = Workflow(name='mapping2struct')
mapping2struct.base_dir = working_dir
mapping2struct.config['execution']['crashdump_dir'] = mapping2struct.base_dir + "/crash_files"

# iterate over subjects
subject_infosource = Node(util.IdentityInterface(fields=['subject']), 
                  name='subject_infosource')
subject_infosource.iterables=[('subject', subjects)]

# iterate over sessions
session_infosource = Node(util.IdentityInterface(fields=['session']), 
                  name='session_infosource')
session_infosource.iterables=[('session', sessions)]

# select files
templates={'mapping': 'mappings/rest/fixed_hdr/corr_{subject}_{session}_roi_detrended_median_corrected_mapping_fixed.nii.gz',
           'epi2highres_lin_itk' : 'resting/preprocessed/{subject}/{session}/registration/epi2highres_lin.txt',
           'epi2highres_warp':'resting/preprocessed/{subject}/{session}/registration/transform0Warp.nii.gz',
           't1_highres' : 'struct/t1/{subject}*T1_Images_merged.nii.gz'
           }    
selectfiles = Node(nio.SelectFiles(templates, base_directory=data_dir),
                   name="selectfiles")

mapping2struct.connect([(subject_infosource, selectfiles, [('subject', 'subject')]),
                 (session_infosource, selectfiles, [('session', 'session')])
                 ])


# merge func2struct transforms into list
translist_forw = Node(util.Merge(2),name='translist_forw')
mapping2struct.connect([(selectfiles, translist_forw, [('epi2highres_lin_itk', 'in2')]),
                 (selectfiles, translist_forw, [('epi2highres_warp', 'in1')])])
   

# project
func2struct = Node(ants.ApplyTransforms(invert_transform_flags=[False, False],
                                        input_image_type=3,
                                        dimension=3,
                                        interpolation='Linear'),
                    name='func2struct')
    
mapping2struct.connect([(selectfiles, func2struct, [('mapping', 'input_image'),
                                                ('t1_highres', 'reference_image')]),
                 (translist_forw, func2struct, [('out', 'transforms')])
                 ])


# split mapping in x,y,z components for sampling to surface
split = Node(fsl.Split(dimension='t'),
             name='split')

mapping2struct.connect([(func2struct, split, [('output_image', 'in_file')])])

  
# sink relevant files
final_sink = Node(nio.DataSink(parameterization=False,
                               base_directory=final_dir),
             name='final_sink')

mapping2struct.connect([(session_infosource, final_sink, [('session', 'container')]),
                   (split, final_sink, [('out_files', '@func')])
                   ])
    
mapping2struct.run(plugin='MultiProc', plugin_args={'n_procs' : 16})
