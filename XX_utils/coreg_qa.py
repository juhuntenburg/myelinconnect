from nilearn import plotting
import pandas as pd
from nipype.interfaces import freesurfer as fs
from glob import glob
import os
import pdb


###settings
wd = '/nobackup/ilz3/myelinconnect/struct/wm_mask/'

subjects = pd.read_csv('/nobackup/ilz3/myelinconnect/subjects.csv')
subjects=list(subjects['DB'])


epi_median = '/nobackup/ilz2/myelinconnect/resting/preprocessed/%s/%s/registration/transform_Warped.nii.gz'
seg = '/nobackup/ilz3/myelinconnect/struct/seg/%s*seg_merged.nii.gz'
wm_mask = '/nobackup/ilz3/myelinconnect/struct/wm_mask/%s*seg_merged_thresh.nii.gz'
pic = '/nobackup/ilz2/myelinconnect/resting/coreg_qa/%s_%s_coreg.png'

sess='rest1_1'# ,'rest1_2', 'rest2_1', 'rest2_2']


create_masks = False

coords = [[10,140,11],[22,88,-69],[25,80,-74], [17,100,-66], [41,84,-26],
          [12,94,-60], [40, 91,-57], [21, 94, -73], [36, 100, -55]]

#pdb.set_trace()
###### run
os.chdir(wd)

if create_masks:
    for sub in subjects:
        masking = fs.Binarize(match = [47,48],
                          out_type = 'nii.gz')
    
        masking.inputs.in_file = glob(seg%sub)[0]
        res = masking.run()
        #wm_mask = res.outputs.binary_file
    
for s in range(len(subjects)):
    
    display = plotting.plot_epi(epi_median%(subjects[s], sess), 
                                cmap='gray', draw_cross = False,
                                annotate = False, cut_coords=coords[s])
    display.add_contours(glob(wm_mask%subjects[s])[0], levels=[.5], colors='r')
    display.savefig(pic%(subjects[s],sess))

    
