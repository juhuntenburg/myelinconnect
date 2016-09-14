import pandas as pd
from nipype.interfaces.freesurfer import Binarize
from nipype.interfaces.io import SelectFiles

subjects = pd.read_csv('/scr/ilz3/myelinconnect/subjects.csv')
subjects=list(subjects['DB'])
subjects.remove('KSMT')


labels= [11, 12, 13, 16, 18] + range(30,42)
templates={'seg': '/scr/ilz3/myelinconnect/struct/seg/{subject}*seg_merged.nii.gz'}
mask_file = '/scr/ilz3/myelinconnect/struct/myelinated_thickness/subcortex_mask/%s_subcortical_mask.nii.gz'


for subject in subjects:
    

    select = SelectFiles(templates)
    select.inputs.subject = subject
    select.run()
    seg_file = select.aggregate_outputs().seg
    
    binarize = Binarize(match = labels,
                        out_type = 'nii.gz')
    binarize.inputs.binary_file = mask_file%subject
    binarize.inputs.in_file=seg_file
    
    binarize.run()
    
    
    
    

