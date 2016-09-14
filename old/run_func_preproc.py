import pandas as pd
from nipype.pipeline.engine import Node, Workflow
import nipype.interfaces.utility as util
import nipype.interfaces.io as nio
import nipype.interfaces.fsl as fsl
from func_preproc.strip_rois import strip_rois_func
from func_preproc.slice_moco import create_slicemoco_pipeline
from func_preproc.twostep_coreg import create_coreg_pipeline
from func_preproc.transform_timeseries import create_transform_pipeline
from func_preproc.denoise import create_denoise_pipeline
from func_preproc.nonlinear import create_nonlinear_pipeline
from nipype.utils.filemanip import list_to_filename

# read in subjects and file names
df=pd.read_csv('/scr/ilz3/myelinconnect/subjects.csv', header=0)
subjects_db=list(df['DB'])
subjects_trt=list(df['TRT'])
subjects=[]
for sub in range(len(subjects_db)):
    subjects.append(subjects_db[sub]+'_'+subjects_trt[sub])
    
t1_files=list(df['T1'])
uni_files=list(df['UNI'])

# sessions to loop over
sessions=['rest1_1' ,'rest1_2', 'rest2_1', 'rest2_2']

# directories
working_dir = '/scr/ilz3/myelinconnect/working_dir/' 
data_dir= '/scr/ilz3/myelinconnect/'
out_dir = '/scr/ilz3/myelinconnect/'
freesurfer_dir = '/scr/ilz3/myelinconnect/freesurfer/'

# set fsl output type to nii.gz
fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

# volumes to remove from each timeseries
vol_to_remove = 5

# bandpass filter cutoffs in HZ, and TR to calculate sigma
TR=3.0
highpass=0.01
lowpass=0.1

# scan parameters for fieldmap correction
#echo_space=0.00033
#te_diff=1.02
#pe_dir='y'


# main workflow
func_preproc = Workflow(name='func_preproc')
func_preproc.base_dir = working_dir
func_preproc.config['execution']['crashdump_dir'] = func_preproc.base_dir + "/crash_files"

# iterate over subjects
subject_infosource = Node(util.IdentityInterface(fields=['subject', 't1', 'uni']), 
                  name='subject_infosource')
subject_infosource.iterables=[('subject', subjects), ('t1', t1_files), ('uni', uni_files)]
subject_infosource.synchronize=True

# iterate over sessions
session_infosource = Node(util.IdentityInterface(fields=['session']), 
                  name='session_infosource')
session_infosource.iterables=[('session', sessions)]

# select files
templates={'rest' : 'resting/raw/{subject}_{session}.nii.gz',
           'dicom':'resting/raw/example_dicoms/{subject}/{session}/*',
           'uni_highres' : 'mp2rage/highres/{uni}',
           't1_highres' : 'mp2rage/highres/{t1}',
           #'brain_mask' : 
           #'uni_lowres' : 'preprocessed/mp2rage/T1_brain.nii.gz',
           #'uni_lowres_brainmask' : 'preprocessed/mp2rage/T1_brain_mask.nii.gz',
           # 'brain_mask' : 'preprocessed/mp2rage/T1_brain_mask.nii.gz',
           #'uni_lowres' : 'freesurfer/{subject}/mri/brain.mgz',
           #'highres2lowres_itk' : 'preprocessed/mp2rage/highres2lowres.txt'
           }    
selectfiles = Node(nio.SelectFiles(templates, base_directory=data_dir),
                   name="selectfiles")

func_preproc.connect([(subject_infosource, selectfiles, [('subject', 'subject'),
                                                        ('t1', 't1'),
                                                        ('uni', 'uni')]),
                      (session_infosource, selectfiles, [('session', 'session')])
                      ])

# node to remove first volumes
remove_vol = Node(util.Function(input_names=['in_file','t_min'],
                                output_names=["out_file"],
                                function=strip_rois_func),
                  name='remove_vol')
remove_vol.inputs.t_min = vol_to_remove

func_preproc.connect([(selectfiles, remove_vol, [('rest', 'in_file')])])

# workflow for slice time and motion correction
slicemoco=create_slicemoco_pipeline()

func_preproc([(selectfiles, slicemoco, [(('dicom',list_to_filename), 'inputnode.dicom')]),
              (remove_vol, slicemoco, [('out_file', 'inputnode.epi')])
              ])


# workflow for linear coregistration
coreg=create_coreg_pipeline()
coreg.inputs.inputnode.fs_subjects_dir = freesurfer_dir

func_preproc([(selectfiles, coreg, [('uni_highres', 'inputnode.uni_highres')]),
              (slicemoco, coreg, [('outputnode.epi_median', 'inputnode.epi_median')]),
              (subject_infosource, coreg, [('subject', 'inputnode.fs_subject_id')])
              ])


# workflow for nonlinear coregistration
# nonreg=create_nonlinear_pipeline()
# 
# func_preproc([(selectfiles, nonreg, [('t1_highres', 'inputnode.t1_highres'),
#                                    ('brain_mask', 'inputnode.brain_mask')]),
#               coreg, nonreg, [('outputnode.epi2highres', 'inputnode.epi2highres'),
#                               ('outputnode.epi2highres_itk', 'inputnode.epi2highres_itk')]),
#             
#                                              
#                                              ('highres2lowres_itk', 'inputnode.highres2lowres_itk'),
#                       (slicemoco, nonreg, [('outputnode.fov_mask', 'inputnode.fov_mask')]),
#                       (coreg, nonreg, [('outputnode.epi2highres', 'inputnode.epi2highres'),
#                                        ('outputnode.epi2highres_itk', 'inputnode.epi2highres_itk')]),
# 
# # # workflow for applying transformations to timeseries
# #transform_ts = create_transform_pipeline()
# #transform_ts.inputs.inputnode.resolution=epi_resolution
# # 
# # # workflow to denoise timeseries
# denoise = create_denoise_pipeline()
# # denoise.inputs.inputnode.highpass=highpass
# # denoise.inputs.inputnode.lowpass=lowpass
# denoise.inputs.inputnode.highpass_sigma= 1./(2*TR*highpass)
# denoise.inputs.inputnode.lowpass_sigma= 1./(2*TR*lowpass)
# # #denoise.inputs.inputnode.resolution = epi_resolution
# #denoise.inputs.inputnode.tr = TR
# # # https://www.jiscmail.ac.uk/cgi-bin/webadmin?A2=ind1205&L=FSL&P=R57592&1=FSL&9=A&I=-3&J=on&d=No+Match%3BMatch%3BMatches&z=4
# 
# 
# #sink to store files
# sink = Node(nio.DataSink(parameterization=False,
#                          base_directory=out_dir,
#                          substitutions=[('transform_Warped.nii.gz', 'epi2highres_nonlin.nii.gz'),
#                                         ('rest_denoised_bp_norm.nii.gz', 'rest_denoise_bandpass_norm.nii.gz')
#                                         ]),
#              name='sink')
# 
# 
# # connections
# func_preproc.connect([#(session_infosource, selectfiles, [('session', 'session')]),
#                       #(session_infosource, sink, [('session', 'container')]),
#                       #(selectfiles, remove_vol, [('func', 'in_file')]),
#                       #(remove_vol, slicetime, [('out_file', 'in_file')]),
#                       #(slicetime, moco, [('out_file', 'inputnode.epi')]),
#                       #(selectfiles, slicemoco, [(('dicom',list_to_filename), 'inputnode.dicom')]),
#                       #(remove_vol, slicemoco, [('out_file', 'inputnode.epi')]),
#                       #(selectfiles, coreg, [('wm_seg', 'inputnode.wmseg'),
#                       #                      ('t1', 'inputnode.anat_brain')
#                       #                      ]),
#                       #(moco, coreg, [('outputnode.epi_mean', 'inputnode.epi_mean')]),
# #                       (selectfiles, coreg, [('uni_lowres', 'inputnode.uni_lowres'),
# #                                             ('uni_highres', 'inputnode.uni_highres'),
# #                                             ('highres2lowres_itk', 'inputnode.highres2lowres_itk')]),
# #                       (slicemoco, coreg, [('outputnode.epi_median', 'inputnode.epi_median')]),
#                       (selectfiles, nonreg, [('t1_highres', 'inputnode.t1_highres'),
#                                              ('highres2lowres_itk', 'inputnode.highres2lowres_itk'),
#                                              ('brain_mask', 'inputnode.brain_mask')]),
#                       (slicemoco, nonreg, [('outputnode.fov_mask', 'inputnode.fov_mask')]),
#                       (coreg, nonreg, [('outputnode.epi2highres', 'inputnode.epi2highres'),
#                                        ('outputnode.epi2highres_itk', 'inputnode.epi2highres_itk')]),
# #                      (remove_vol, transform_ts, [('out_file', 'inputnode.orig_ts')]),
# #                        (slicemoco, transform_ts, [('outputnode.epi_moco', 'inputnode.epi_moco')]),
# #                        (selectfiles, transform_ts, [('t1_highres', 'inputnode.t1_highres')]),
# # # #                     (moco, transform_ts, [('outputnode.mat_moco', 'inputnode.mat_moco'),
# # # #                                           ('outputnode.epi_mean', 'inputnode.epi_mean')]),
# #                        (coreg, transform_ts, [('outputnode.epi2highres_itk', 'inputnode.epi2highres_itk')]),
# #                        (nonreg, transform_ts, [('outputnode.epi2anat_transforms', 'inputnode.epi2highres_nonlin')]),
# # # #                     (selectfiles, denoise, [('brain_seg', 'inputnode.brain_seg')]),
# # #                      (moco, denoise, [('outputnode.par_moco', 'inputnode.motion_parameters')]),
# #                        (selectfiles, denoise, [('uni_lowres', 'inputnode.uni_lowres_brain'),
# #                                                ('uni_lowres_brainmask', 'inputnode.uni_lowres_brainmask'),
# #                                                ('highres2lowres_itk', 'inputnode.highres2lowres_itk')]),
# #                        (transform_ts, denoise, [('outputnode.trans_ts','inputnode.epi_coreg'),
# #                                                 ('outputnode.t1_highres_resamp','inputnode.t1_highres_resamp')]),
# #                        (slicemoco, denoise, [('outputnode.par_moco', 'inputnode.moco_par'),
# #                                              ('outputnode.tr', 'inputnode.tr')]),
# # #                       (moco, sink, [('outputnode.epi_moco', 'realign.@realigned_ts'),
# # #                                     ('outputnode.fov_mask', 'realign.@fov_mask'),
# # #                                     ('outputnode.par_moco', 'realign.@par'),
# # #                                     ('outputnode.rms_moco', 'realign.@rms'),
# # #                                     ('outputnode.mat_moco', 'realign.MAT.@mat'),
# # #                                     ('outputnode.epi_mean', 'realign.@mean'),
# # #                                     ('outputnode.rotplot', 'realign.plots.@rotplot'),
# # #                                     ('outputnode.transplot', 'realign.plots.@transplot'),
# # #                                     ('outputnode.dispplots', 'realign.plots.@dispplots')]),
# # 
# #                       (slicemoco, sink, [('outputnode.epi_moco', 'realign.@realigned_ts'),
# #                                          ('outputnode.fov_mask', 'realign.@fov_mask'),
# #                                          ('outputnode.par_moco', 'realign.@par'),
# #                                          ('outputnode.epi_median', 'realign.@median')]),
# #                       (coreg, sink, [('outputnode.epi2lowres', 'coregister.@coreg_mean'),
# #                                      ('outputnode.epi2lowres_mat', 'coregister.@epi2anat_mat'),
# #                                      ('outputnode.epi2lowres_dat', 'coregister.@epi2anat_dat'),
# #                                      ('outputnode.epi2highres', 'coregister.@epi2highres' ),
# #                                      ('outputnode.epi2highres_itk', 'coregister.@epi2highres_itk')]),
# #                       (nonreg, sink, [('outputnode.epi2anat', 'coregister.@nonreg_epi2anat'),
# #                                       ('outputnode.epi2anat_transforms', 'coregister.@nonreg_epi2anat_transforms'),
# #                                       ('outputnode.anat2epi_transforms', 'coregister.@nonreg_anat2epi_transforms')]),
# #                        (transform_ts, sink, [('outputnode.trans_ts', 'coregister.@full_transform_ts'),
# #                                              ('outputnode.trans_ts_mean', 'coregister.@full_transform_mean'),
# #                                              ('outputnode.t1_highres_resamp', 'coregister.@t1_resamp')]),
# #                     (denoise, sink, [('outputnode.wmcsf_mask', 'denoise.@wmcsf_masks'),
# #                                      ('outputnode.brain_mask_resamp', 'denoise.@brain'),
# #                                       ('outputnode.combined_motion','denoise.noise.@combined_motion'),
# #                                       ('outputnode.outlier_files','denoise.noise.@outlier'),
# #                                       ('outputnode.intensity_files','denoise.noise.@intensity'),
# #                                       ('outputnode.outlier_stats','denoise.noise.@outlierstats'),
# #                                       ('outputnode.outlier_plots','denoise.@outlierplots'),
# #                                       ('outputnode.mc_regressor', 'denoise.noise.@mc_regressor'),
# #                                       ('outputnode.comp_regressor', 'denoise.noise.@comp_regressor'),
# #                                       ('outputnode.mc_F', 'denoise.noise.@mc_F'),
# #                                       ('outputnode.mc_pF', 'denoise.noise.@mc_pF'),
# #                                       ('outputnode.comp_F', 'denoise.noise.@comp_F'),
# #                                       ('outputnode.comp_pF', 'denoise.noise.@comp_pF'),
# #                                       ('outputnode.normalized_file', '@normalized')])
# #])
# 
# #joinnode for concatenation
# # concatenate=JoinNode(fsl.Merge(dimension='t',
# #                            merged_file='rest_concatenated.nii.gz'),
# #                      joinsource='session_infosource',
# #                      joinfield='in_files',
# #                      name='concatenate')
# # concatenate.plugin_args={'submit_specs': 'request_memory = 20000'}
# #  
# # concat_sink=Node(nio.DataSink(parameterization=False,
# #                               base_directory=out_dir),
# #                  name='concat_sink')
# #  
# #  
# # func_preproc.connect([(denoise, concatenate, [('outputnode.normalized_file', 'in_files')]),
# #                       (concatenate, concat_sink, [('merged_file', '@rest_concat')])])
# 
# 
# 
# #func_preproc.write_graph(dotfilename='func_preproc.dot', graph2use='colored', format='pdf', simple_form=True)
func_preproc.run()