from nipype.pipeline.engine import Node, Workflow
import nipype.interfaces.io as nio
from struct_preproc.mp2rage import create_mp2rage_pipeline
from struct_preproc.reconall import create_reconall_pipeline
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.c3 as c3
from nipype.interfaces.mipav.developer import JistIntensityMp2rageMasking
import nipype.interfaces.freesurfer as fs


def create_structural(subject, working_dir, data_dir, freesurfer_dir, out_dir):
    
    '''
    Workflow to run brackground masking and then freesurfer recon-all
    on "lowres" MP2RAGE data
    '''
    
    # main workflow
    struct_preproc = Workflow(name='mp2rage_preproc')
    struct_preproc.base_dir = working_dir
    struct_preproc.config['execution']['crashdump_dir'] = struct_preproc.base_dir + "/crash_files"
    
    # select files
    templates={'inv2': 'raw/mp2rage/inv2.nii.gz',
               't1map': 'raw/mp2rage/t1map.nii.gz',
               'uni': 'raw/mp2rage/uni.nii.gz'}
    selectfiles = Node(nio.SelectFiles(templates,
                                       base_directory=data_dir),
                       name="selectfiles")
    
    # mp2rage background masking
    background = Node(JistIntensityMp2rageMasking(outMasked=True,
                                            outMasked2=True,
                                            outSignal2=True), 
                      name='background')
    
    
    
    # workflow to run freesurfer reconall
    
    # function to replace / in subject id string with a _
    def sub_id(sub_id):
        return sub_id.replace('/','_')
    
    recon_all = Node(fs.ReconAll(args='-nuiterations 7 -no-isrunning'),
                     name="recon_all")
    recon_all.plugin_args={'submit_specs': 'request_memory = 9000'}
    recon_all.inputs.subjects_dir=freesurfer_dir
    recon_all.inputs.subject_id=sub_id(subject)
    
    
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
    struct_preproc.connect([(selectfiles, background, [('inv2', 'inSecond'),
                                                       ('t1map', 'inQuantitative'),
                                                       ('uni', 'inT1weighted')]),
                            (background, recon_all, [('outMasked2','T1files')]),
                            (background, sink, [('outMasked2','preprocessed.mp2rage.@uni_masked'),
                                                ('outSignal2','preprocessed.mp2rage.@background_mask')]),
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
struct.run()
