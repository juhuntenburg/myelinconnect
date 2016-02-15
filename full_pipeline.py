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
# subjects_trt=list(df['TRT'])
# subjects=[]
# for sub in range(len(subjects_db)):
#     subjects.append(subjects_db[sub]+'_'+subjects_trt[sub])
subjects=['KSYT', 'WSFT']
# sessions to loop over
sessions=['rest1_1' ,'rest1_2', 'rest2_1', 'rest2_2']

# directories
working_dir = '/scr/ilz3/myelinconnect/working_dir/' 
data_dir= '/scr/ilz3/myelinconnect/'
out_dir = '/scr/ilz3/myelinconnect/resting/preprocessed/'
freesurfer_dir = '/scr/ilz3/myelinconnect/freesurfer/' # freesurfer reconstruction of lowres is assumed

# set fsl output type to nii.gz
fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

# volumes to remove from each timeseries
vol_to_remove = 5

# main workflow
preproc = Workflow(name='func_preproc')
preproc.base_dir = working_dir
preproc.config['execution']['crashdump_dir'] = preproc.base_dir + "/crash_files"

# iterate over subjects
subject_infosource = Node(util.IdentityInterface(fields=['subject']), 
                  name='subject_infosource')
subject_infosource.iterables=[('subject', subjects_db)]

# iterate over sessions
session_infosource = Node(util.IdentityInterface(fields=['session']), 
                  name='session_infosource')
session_infosource.iterables=[('session', sessions)]

# select files
templates={'rest' : 'resting/raw/{subject}_{session}.nii.gz',
           'dicom':'resting/raw/example_dicoms/{subject}*/{session}/*',
           'uni_highres' : 'struct/uni/{subject}*UNI_Images_merged.nii.gz',
           't1_highres' : 'struct/t1/{subject}*T1_Images_merged.nii.gz',
           'brain_mask' : 'struct/mask/{subject}*mask.nii.gz',
           'segmentation' : 'struct/seg/{subject}*lbls_merged.nii.gz',
           'csfmask' : 'struct/csfmask/{subject}*T1_Images_merged_seg_merged_sub_csf.nii.gz'
           }    
selectfiles = Node(nio.SelectFiles(templates, base_directory=data_dir),
                   name="selectfiles")

preproc.connect([(subject_infosource, selectfiles, [('subject', 'subject')]),
                 (session_infosource, selectfiles, [('session', 'session')])
                 ])

# remove first volumes
remove_vol = Node(util.Function(input_names=['in_file','t_min'],
                                output_names=["out_file"],
                                function=strip_rois_func),
                  name='remove_vol')
remove_vol.inputs.t_min = vol_to_remove

preproc.connect([(selectfiles, remove_vol, [('rest', 'in_file')])])

# get slice time information from example dicom
getinfo = Node(util.Function(input_names=['dicom_file'],
                             output_names=['TR', 'slice_times', 'slice_thickness'],
                             function=get_info),
               name='getinfo')
preproc.connect([(selectfiles, getinfo, [('dicom', 'dicom_file')])])
                 
                 
# simultaneous slice time and motion correction
slicemoco = Node(nipy.SpaceTimeRealigner(),                 
                 name="spacetime_realign")
slicemoco.inputs.slice_info = 2

preproc.connect([(getinfo, slicemoco, [('slice_times', 'slice_times'),
                                       ('TR', 'tr')]),
                 (remove_vol, slicemoco, [('out_file', 'in_file')])])

# compute tsnr and detrend
tsnr = Node(TSNR(regress_poly=2),
               name='tsnr')
preproc.connect([(slicemoco, tsnr, [('out_file', 'in_file')])])
 
# compute median of realigned timeseries for coregistration to anatomy
median = Node(util.Function(input_names=['in_files'],
                       output_names=['median_file'],
                       function=median),
              name='median')
 
preproc.connect([(tsnr, median, [('detrended_file', 'in_files')])])
 
# make FOV mask for later nonlinear coregistration
fov = Node(fsl.maths.MathsCommand(args='-bin',
                                  out_file='fov_mask.nii.gz'),
           name='fov_mask')
preproc.connect([(median, fov, [('median_file', 'in_file')])])

# fix header of brain mask
fixhdr = Node(util.Function(input_names=['data_file', 'header_file'],
                            output_names=['out_file'],
                            function=fix_hdr),
                  name='fixhdr')
preproc.connect([(selectfiles, fixhdr, [('brain_mask', 'data_file'),
                                        ('t1_highres', 'header_file')]),
                 ])

# biasfield correction of median epi for better registration
biasfield = Node(ants.segmentation.N4BiasFieldCorrection(save_bias=True),
                 name='biasfield')
preproc.connect([(median, biasfield, [('median_file', 'input_image')])])

# perform linear coregistration in two steps, median2lowres, lowres2highres
coreg=create_coreg_pipeline()
coreg.inputs.inputnode.fs_subjects_dir = freesurfer_dir
 
preproc.connect([(selectfiles, coreg, [('uni_highres', 'inputnode.uni_highres')]),
                 (biasfield, coreg, [('output_image', 'inputnode.epi_median')]),
                 (subject_infosource, coreg, [('subject', 'inputnode.fs_subject_id')])
                 ])

# perform nonlinear coregistration 
nonreg=create_nonlinear_pipeline()
   
preproc.connect([(selectfiles, nonreg, [('t1_highres', 'inputnode.t1_highres')]),
                 (fixhdr, nonreg, [('out_file', 'inputnode.brain_mask')]),
                 (fov, nonreg, [('out_file', 'inputnode.fov_mask')]),
                 (coreg, nonreg, [('outputnode.epi2highres_lin', 'inputnode.epi2highres_lin'),
                                  ('outputnode.epi2highres_lin_itk', 'inputnode.epi2highres_lin_itk')])
                  ])

# make wm/csf mask from segmentations and erode, and medial wall mask
wmmask = Node(fs.Binarize(match = [46,47,48],
                          erode = 4,
                          out_type = 'nii.gz',
                          binary_file='wm_mask.nii.gz'), 
               name='wmmask')


csfmask = Node(fs.Binarize(match=[1],
                          erode = 2,
                          out_type = 'nii.gz',
                          binary_file='csf_mask.nii.gz'), 
               name='csfmask')

preproc.connect([(selectfiles, wmmask, [('segmentation', 'in_file')]),
                 (selectfiles, csfmask, [('csfmask', 'in_file')])
                 ])

# merge struct2func transforms into list
translist_inv = Node(util.Merge(2),name='translist_inv')
preproc.connect([(coreg, translist_inv, [('outputnode.epi2highres_lin_itk', 'in1')]),
                 (nonreg, translist_inv, [('outputnode.epi2highres_invwarp', 'in2')])])
   
# merge images into list
structlist = Node(util.Merge(3),name='structlist')
preproc.connect([(fixhdr, structlist, [('out_file', 'in1')]),
                 (wmmask, structlist, [('binary_file', 'in2')]),
                 (csfmask, structlist, [('binary_file', 'in3')])
                 ])
   
# project brain mask, wm/csf masks, t1 and subcortical mask in functional space
struct2func = MapNode(ants.ApplyTransforms(dimension=3,
                                         invert_transform_flags=[True, False],
                                         interpolation = 'NearestNeighbor'),
                    iterfield=['input_image'],
                    name='struct2func')

   
preproc.connect([(structlist, struct2func, [('out', 'input_image')]),
                 (translist_inv, struct2func, [('out', 'transforms')]),
                 (median, struct2func, [('median_file', 'reference_image')]),
                 ])


# calculate compcor regressors
compcor = Node(util.Function(input_names=['realigned_file', 'mask_file',
                                          'num_components',
                                          'extra_regressors'],
                                   output_names=['out_files'],
                                   function=extract_noise_components),
                     name='compcor')
compcor.inputs.num_components = 5
preproc.connect([(slicemoco, compcor, [('out_file', 'realigned_file')]),
                 (struct2func, compcor, [(('output_image', selectindex, [1,2]),'mask_file')])
                 ])
  
# perform artefact detection
artefact=Node(ra.ArtifactDetect(save_plot=True,
                                use_norm=True,
                                parameter_source='NiPy',
                                mask_type='file',
                                norm_threshold=1,
                                zintensity_threshold=3,
                                use_differences=[True,False]
                                ),
             name='artefact')
   
preproc.connect([(slicemoco, artefact, [('out_file', 'realigned_files'),
                                        ('par_file', 'realignment_parameters')]),
                 (struct2func, artefact, [(('output_image', selectindex, [0]), 'mask_file')]),
                 ])
  
  
# calculate motion regressors
motreg = MapNode(util.Function(input_names=['motion_params', 'order','derivatives'],
                            output_names=['out_files'],
                            function=motion_regressors),
                 iterfield=['order'],
                 name='motion_regressors')
motreg.inputs.order=[1,2]
motreg.inputs.derivatives=1
preproc.connect([(slicemoco, motreg, [('par_file','motion_params')])])
  
def makebase(subject, out_dir):
    return out_dir + subject
  
# sink relevant files
sink = Node(nio.DataSink(parameterization=False),
             name='sink')
  
preproc.connect([(session_infosource, sink, [('session', 'container')]),
                 (subject_infosource, sink, [(('subject', makebase, out_dir), 'base_directory')]),
                 (remove_vol, sink, [('out_file', 'realignment.@raw_file')]),
                 (slicemoco, sink, [('out_file', 'realignment.@realigned_file'),
                                    ('par_file', 'confounds.@orig_motion')]),
                 (tsnr, sink, [('tsnr_file', 'realignment.@tsnr')]),
                 (median, sink, [('median_file', 'realignment.@median')]),
                 (biasfield, sink, [('output_image', 'realignment.@biasfield')]),
                 (coreg, sink, [('outputnode.uni_lowres', 'registration.@uni_lowres'),
                                ('outputnode.epi2lowres', 'registration.@epi2lowres'),
                                ('outputnode.epi2lowres_mat','registration.@epi2lowres_mat'),
                                ('outputnode.epi2lowres_dat','registration.@epi2lowres_dat'),
                                ('outputnode.highres2lowres', 'registration.@highres2lowres'),
                                ('outputnode.highres2lowres_dat', 'registration.@highres2lowres_dat'),
                                ('outputnode.highres2lowres_mat', 'registration.@highres2lowres_mat'),
                                ('outputnode.epi2highres_lin', 'registration.@epi2highres_lin'),
                                ('outputnode.epi2highres_lin_itk', 'registration.@epi2highres_lin_itk'),
                                ('outputnode.epi2highres_lin_mat', 'registration.@epi2highres_lin_mat')]),
                (nonreg, sink, [('outputnode.epi2highres_warp', 'registration.@epi2highres_warp'),
                                ('outputnode.epi2highres_invwarp', 'registration.@epi2highres_invwarp'),
                                ('outputnode.epi2highres_nonlin', 'registration.@epi2highres_nonlin')]),
                (struct2func, sink, [(('output_image', selectindex, [0,1,2]), 'mask.@masks')]),
                (artefact, sink, [('norm_files', 'confounds.@norm_motion'),
                                  ('outlier_files', 'confounds.@outlier_files'),
                                  ('intensity_files', 'confounds.@intensity_files'),
                                  ('statistic_files', 'confounds.@outlier_stats'),
                                  ('plot_files', 'confounds.@outlier_plots')]),
                 (compcor, sink, [('out_files', 'confounds.@compcor')]),
                 (motreg, sink, [('out_files', 'confounds.@motreg')])
                 ])
    
#preproc.run(plugin='MultiProc', plugin_args={'n_procs' : 9})

preproc.write_graph(dotfilename='func_preproc.dot', graph2use='colored', format='pdf', simple_form=True)
