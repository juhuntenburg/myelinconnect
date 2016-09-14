# imports for pipeline
from nipype.pipeline.engine import Node, Workflow
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
import nipype.interfaces.io as nio

subject_id = raw_input('subject ID: ')

# local base and output directory (should usually not change)
data_dir = '/scr/ilz3/7tresting/'+subject_id+'/preprocessed/'
base_dir = '/scr/ilz3/7tresting/working_dir/'+subject_id+'/'
out_dir = '/scr/ilz3/7tresting/'+subject_id+'/preprocessed/'

# workflow
concat = Workflow(name='concat')
concat.base_dir = base_dir
concat.config['execution']['crashdump_dir'] = concat.base_dir + "/crash_files"

# select files
templates={'rest1_1': 'rest1_1/rest_denoise_bandpass_norm.nii.gz',
           'rest1_2': 'rest1_2/rest_denoise_bandpass_norm.nii.gz',
           'rest2_1': 'rest2_1/rest_denoise_bandpass_norm.nii.gz',
           'rest2_2': 'rest2_2/rest_denoise_bandpass_norm.nii.gz',
           }
selectfiles = Node(nio.SelectFiles(templates,
                                   base_directory=data_dir),
                   name="selectfiles")

# make filelist
def makelist(in1, in2, in3, in4):
    return [in1, in2, in3, in4]

make_list = Node(util.Function(input_names=['in1', 'in2', 'in3', 'in4'],
                               output_names=['file_list'],
                               function=makelist),
                               name='make_list')

# concatenate scans
concatenate=Node(fsl.Merge(dimension='t',
                           merged_file='rest_preprocessed_concat.nii.gz'),
                 name='concatenate')
concatenate.plugin_args={'submit_specs': 'request_memory = 20000'}

# sink
sink = Node(nio.DataSink(base_directory=out_dir,
                         parameterization=False),
                name='sink')

concat.connect([(selectfiles, make_list, [('rest1_1', 'in1'),
                                          ('rest1_2', 'in2'),
                                          ('rest2_1', 'in3'),
                                          ('rest2_2', 'in4')]),
                (make_list, concatenate, [('file_list', 'in_files')]),
                (concatenate, sink, [('merged_file', '@rest_concat')])
                ])

concat.run()#plugin='CondorDAGMan')