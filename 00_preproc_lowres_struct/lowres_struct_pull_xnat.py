# imports for pipeline
from nipype.pipeline.engine import Node, Workflow
import nipype.interfaces.utility as util
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

# xnat configuration for pipeline source data 
project_id = 'VAR7T_dataset'
subject_id = 'sub021'
exp_id = subject_id+'_Session1_defacednii'
 

# local base and output directory 
base_dir = '/scr/ilz2/7tresting/working_dir/' +subject_id+'/' 
out_dir = '/scr/ilz2/7tresting/'

# scans to be pulled
scans=dict()
scans['inv1']='MP2RAGE_INV1'
scans['inv2']='MP2RAGE_INV2'
scans['t1map']='MP2RAGE_T1'
scans['uni']='MP2RAGE_UNI'

pull = Workflow(name='pull')
pull.base_dir = base_dir
pull.config['execution']['crashdump_dir'] = pull.base_dir + "/crash_files"

# infosource to iterate over scans
scan_infosource = Node(util.IdentityInterface(fields=['scan_key', 'scan_val']), 
                  name='scan_infosource')
scan_infosource.iterables=[('scan_key', scans.keys()), ('scan_val', scans.values())]
scan_infosource.synchronize=True

# xnat source
xnatsource = Node(nio.XNATSource(infields=['project_id', 'subject_id', 
                                       'exp_id', 'scan_id'],
                             outfields=['nifti'],
                             server=xnat_server,
                             user=xnat_user,
                             pwd=xnat_pass, 
                             cache_dir=base_dir),
          name='xnatsource')

xnatsource.inputs.query_template=('/projects/%s/subjects/%s/experiments/%s/scans/%s/resources/NIFTI/files')#files')
xnatsource.inputs.query_template_args['nifti']=[['project_id', 'subject_id', 'exp_id', 'scan_id']]
xnatsource.inputs.project_id = project_id
xnatsource.inputs.subject_id = subject_id
xnatsource.inputs.exp_id = exp_id
pull.connect([(scan_infosource, xnatsource, [('scan_val', 'scan_id')])])


# rename files
rename=Node(util.Rename(format_string="%(scan_key)s.nii.gz",
                        keep_ext=False),
            name='rename')

pull.connect([(xnatsource, rename, [('nifti', 'in_file')]),
             (scan_infosource, rename, [('scan_key', 'scan_key')])
             ])
             
    

# xnat sink
sink = Node(nio.DataSink(base_directory=out_dir,
                         parameterization=False,
                         container=subject_id[0:7]),
                name='sink')


pull.connect([(rename, sink, [('out_file', 'raw.mp2rage.@nifti')])])

pull.run()#(plugin='CondorDAGMan')


#dcmstack --dest-dir /scr/kansas1/huntenburg/7tresting/sub021/nifti/mp2rage/ -o uni /scr/kalifornien1/7T_TRT/test_data/tmp_dicoms/sub021/session_1/MP2RAGE_UNI/