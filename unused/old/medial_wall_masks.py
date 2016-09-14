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

# directories
working_dir = '/scr/ilz3/myelinconnect/working_dir/' 
data_dir= '/scr/ilz3/myelinconnect/struct/'
out_dir = '/scr/ilz3/myelinconnect/struct/medial_wall'
# set fsl output type to nii.gz
fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

# main workflow
medwall = Workflow(name='medwall')
medwall.base_dir = working_dir
medwall.config['execution']['crashdump_dir'] = medwall.base_dir + "/crash_files"

# iterate over subjects
subject_infosource = Node(util.IdentityInterface(fields=['subject']), 
                  name='subject_infosource')
subject_infosource.iterables=[('subject', subjects_db)]

# select files
templates={'sub_gm': 'medial_wall/input/{subject}*_sub_gm.nii.gz',
           'sub_csf': 'medial_wall/input/{subject}*_sub_csf.nii.gz',
           'thickness_rh' : 'thickness/{subject}*right_cortex_thick.nii.gz',
           'thickness_lh' : 'thickness/{subject}*left_cortex_thick.nii.gz',}

selectfiles = Node(nio.SelectFiles(templates, base_directory=data_dir),
                   name="selectfiles")

medwall.connect([(subject_infosource, selectfiles, [('subject', 'subject')])
                 ])

addmasks= Node(fsl.BinaryMaths(operation='add'),
               name='addmasks')

medwall.connect([(selectfiles, addmasks, [('sub_gm', 'in_file'),
                                          ('sub_csf', 'operand_file')])])

morph_closing = Node(fs.Binarize(min=0.5,
                                 dilate=10,
                                 erode=10),
                     name='morph_close')

medwall.connect([(addmasks, morph_closing, [('out_file', 'in_file')])])



'''alternative with thickness'''
wallmask_rh = Node(fs.Binarize(max=0.2,
                            out_type = 'nii.gz'), 
               name='wallmask_rh')
  
wallmask_lh = wallmask_rh.clone('wallmask_lh')
 
medwall.connect([(selectfiles, wallmask_rh, [('thickness_rh', 'in_file')]),
                 (selectfiles, wallmask_lh, [('thickness_lh', 'in_file')])
                 ])

addmasks2= Node(fsl.BinaryMaths(operation='add'),
               name='addmasks2')

medwall.connect([(wallmask_rh, addmasks2, [('binary_file', 'in_file')]),
                 (wallmask_lh, addmasks2, [('binary_file', 'operand_file')])])

'''
followed by
3dclust -savemask $out 0 20 $in
'''
  
# sink relevant files
sink = Node(nio.DataSink(parameterization=False,
                               base_directory=out_dir),
             name='sink')

medwall.connect([(morph_closing, sink, [('binary_file', '@fullmask')]),
                 (addmasks2, sink, [('out_file', '@alternativemask')])
                 ])

medwall.run(plugin='MultiProc', plugin_args={'n_procs' : 9})
