from __future__ import division
import numpy as np
from functions import fix_hdr
import pandas as pd
import os
from joblib import Parallel, delayed



'''
Overwrites headers of mapping derived from functional median file
with header from this original median file (mipav screws them up).
Only after this applying the transformations to the mapping will
result in the correct mapping.
'''

# function returning generator for sub,rest tuples fro joblib
def tupler(subjects, rests):
    for s in subjects:
        for r in rests:
            yield (s, r)
            
            
def loop_fix_hdr((sub, rest)):
    
    mapping_file = '/scr/ilz3/myelinconnect/mappings/rest/orig_hdr/corr_%s_rest%s_roi_detrended_median_corrected_mapping.nii.gz'%(sub, rest)
    median_file =  '/scr/ilz3/myelinconnect/resting/preprocessed/%s/rest%s/realignment/corr_%s_rest%s_roi_detrended_median_corrected.nii.gz'%(sub, rest, sub, rest)
    out_dir = '/scr/ilz3/myelinconnect/mappings/rest/fixed_hdr/'
    
    os.chdir(out_dir)
    fixed_file = fix_hdr(mapping_file, median_file)
    
    return fixed_file

if __name__ == "__main__":

    subjects = pd.read_csv('/scr/ilz3/myelinconnect/subjects.csv')
    subjects=list(subjects['DB'])
    subjects.remove('KSMT')
    
    rests = ['1_1', '1_2', '2_1', '2_2']

    Parallel(n_jobs=16)(delayed(loop_fix_hdr)(i) 
                               for i in tupler(subjects, rests))
    
