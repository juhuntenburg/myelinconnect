import numpy as np
import pandas as pd

subjects = pd.read_csv('/scr/ilz3/myelinconnect/subjects.csv')
subjects=list(subjects['DB'])
#subjects.remove('KSMT')

scans = ['rest1_1', 'rest1_2', 'rest2_1', 'rest2_2']

composite_norm_file = '/scr/ilz2/myelinconnect/resting/preprocessed/%s/%s/confounds/norm.corr_%s_%s_roi.txt'
composite_all_file = '/scr/ilz2/myelinconnect/resting/composite_motion_all_subs.csv'

df = pd.DataFrame(index=['max_all', 'mean_all', '>.5', '>1', '>2', '>3', 
                         'max_1', 'mean_1',
                         'max_2', 'mean_2',
                         'max_3', 'mean_3',
                         'max_4', 'mean_4', ], columns=subjects)

for sub in subjects: 
    norm = np.zeros((295*4,))
    for s in range(4):
        scan = scans[s]
        norm[s*295:(s+1)*295] = np.loadtxt(composite_norm_file %(sub, scan, sub, scan))
        
        df[sub] = [np.max(norm), np.mean(norm), 
                   np.where(norm>0.5)[0].shape[0], np.where(norm>1)[0].shape[0], 
                   np.where(norm>2)[0].shape[0], np.where(norm>3)[0].shape[0],
                   np.max(norm[0*295:1*295]), np.mean(norm[0*295:1*295]),
                   np.max(norm[1*295:2*295]), np.mean(norm[1*295:2*295]),
                   np.max(norm[2*295:3*295]), np.mean(norm[2*295:3*295]),
                   np.max(norm[3*295:4*295]), np.mean(norm[3*295:4*295])
                   ]

df.to_csv(composite_all_file)
