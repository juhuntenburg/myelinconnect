# imports for pipeline
from nipype.pipeline.engine import Node, Workflow
import nipype.interfaces.utility as util
from dcmconvert import create_dcmconvert_pipeline
import nipype.interfaces.io as nio
import getpass


'''
Settings and inputs
===============================================================================
'''
# xnat credentials, from command line input
xnat_server = 'https://xnat.cbs.mpg.de/xnat'
xnat_user = raw_input('XNAT username: ')
xnat_pass = getpass.getpass('XNAT password: ')

#subjects=raw_input('subject_id (seperate multiple by space): ')
#subjects=map(str, subjects.split())
subjects=['sub021']#, 'sub014'] #'sub013' exlcude session 1 fmap 1
project_id = 'VAR7T_dataset'

# local base and output directory 
base_dir = '/scr/ilz2/7tresting/working_dir/'
out_dir = '/scr/ilz2/7tresting/'

# scans to be converted
# check if scan assignments match (session1_1 through 2_2)
scan_ids=[12,16,6,15]
scan_names=['rest']*4
sessions=['Session1']*2+['Session2']*2
folders=['rest1_1','rest1_2','rest2_1','rest2_2']
#scan_ids=[10,11,12,14,15,16,4,5,6,11,12,13]
#scan_names=['fmap_mag','fmap_phase','rest']*4
#sessions=['Session1']*6+['Session2']*6
#folders=['rest1_1']*3+['rest1_2']*3+['rest2_1']*3+['rest2_2']*3


'''
Main workflow 
===============================================================================
'''

convert = Workflow(name='convert')
convert.base_dir = base_dir
convert.config['execution']['crashdump_dir'] = convert.base_dir + "/crash_files"

# iterate over subjects
subject_infosource = Node(util.IdentityInterface(fields=['subject_id']), 
                  name='subject_infosource')
subject_infosource.iterables=[('subject_id', subjects)]

# iterate over scans
scan_infosource = Node(util.IdentityInterface(fields=['scan_id', 'scan_name','session','folder']), 
                  name='scan_infosource')
scan_infosource.iterables=[('scan_id', scan_ids), ('scan_name', scan_names),
                           ('session', sessions), ('folder', folders)]
scan_infosource.synchronize=True


# xnat source
xnatsource = Node(nio.XNATSource(infields=['project_id', 'subject_id',
                                           'exp_id', 'scan_id'],
                                 outfields=['dicom'],
                                 server=xnat_server,
                                 user=xnat_user,
                                 pwd=xnat_pass,
                                 cache_dir=base_dir),
                  name='xnatsource')

xnatsource.inputs.query_template=('/projects/%s/subjects/%s/experiments/%s_%s/scans/%d/resources/DICOM/files')
xnatsource.inputs.query_template_args['dicom']=[['project_id', 'subject_id', 'subject_id', 'exp_id', 'scan_id']]
xnatsource.inputs.project_id = project_id
convert.connect([(subject_infosource, xnatsource, [('subject_id','subject_id')]),
                 (scan_infosource, xnatsource, [('scan_id', 'scan_id'),
                                                ('session', 'exp_id')])])


# workflow to convert dicoms
dcmconvert=create_dcmconvert_pipeline()
convert.connect([(scan_infosource, dcmconvert, [('scan_name', 'inputnode.filename')]),
                 (xnatsource, dcmconvert, [('dicom', 'inputnode.dicoms')])])


# function to pich one of the dicoms and save it
def pickdicom(dicomlist):
    example_dcm = dicomlist[5]
    return example_dcm

# make base directory including subject id
def makebase(subject_id, out_dir):
    base_dir=out_dir+subject_id+'/raw/'
    return base_dir

make_base= Node(util.Function(input_names=['subject_id', 'out_dir'],
                              output_names=['base_dir'],
                              function=makebase),
                name='make_base')

make_base.inputs.out_dir=out_dir
convert.connect([(subject_infosource, make_base, [('subject_id', 'subject_id')])])
  
    
# sink
sink = Node(nio.DataSink(base_directory=out_dir,
                         parameterization=False,),
                name='sink')

convert.connect([(make_base, sink, [('base_dir', 'base_directory')]),
                 (scan_infosource, sink, [('folder', 'container')]),
                 (xnatsource, sink, [(('dicom', pickdicom), 'example_dicom.@dicom')]),
                 (dcmconvert, sink, [('outputnode.nifti', '@nifti')])])

convert.run()#(plugin='CondorDAGMan')