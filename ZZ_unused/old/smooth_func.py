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
out_dir = '/scr/ilz3/myelinconnect/final_struct_space/rest1_1_trans'

# set fsl output type to nii.gz
fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

# main workflow
smooth = Workflow(name='smooth')
smooth.base_dir = working_dir
smooth.config['execution']['crashdump_dir'] = smooth.base_dir + "/crash_files"

# iterate over subjects
subject_infosource = Node(util.IdentityInterface(fields=['subject']), 
                  name='subject_infosource')
subject_infosource.iterables=[('subject', subjects_db)]

# iterate over sessions
session_infosource = Node(util.IdentityInterface(fields=['session']), 
                  name='session_infosource')
session_infosource.iterables=[('session', sessions)]

# select files
templates={'rest': 'final_struct_space/rest1_1_trans/{subject}_{session}_denoised_trans.nii.gz'
           }    
selectfiles = Node(nio.SelectFiles(templates, base_directory=data_dir),
                   name="selectfiles")

smooth.connect([(subject_infosource, selectfiles, [('subject', 'subject')]),
                 (session_infosource, selectfiles, [('session', 'session')])
                 ])


# smooth by 3 mm
blur = Node(afni.Merge(blurfwhm=3,
                       doall=True,
                       outputtype='NIFTI_GZ'),
              name='blur')

smooth.connect([(selectfiles, blur, [('rest', 'in_file')])])

  
# sink relevant files
sink = Node(nio.DataSink(parameterization=False,
                               base_directory=out_dir),
             name='sink')

smooth.connect([(blur, sink, [('out_file', '@blur')])
                   ])
    
smooth.run(plugin='MultiProc', plugin_args={'n_procs' : 9})