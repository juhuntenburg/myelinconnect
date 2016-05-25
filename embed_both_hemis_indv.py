from __future__ import division
import numpy as np
import numexpr as ne
import pandas as pd
from correlations import avg_correlation
from clustering import embedding
import h5py
import pickle

ne.set_num_threads(ne.ncores-1)

subjects = pd.read_csv('/scr/ilz3/myelinconnect/subjects.csv')
subjects=list(subjects['DB'])
subjects.remove('KSMT')

smooths=['smooth_3'] 
sessions = ['1_1', '1_2']# , '2_1', '2_2']
n_embedding = 100

rest_file = '/scr/ilz3/myelinconnect/new_groupavg/rest/smooth_3/%s_%s_rest%s_smooth_3.npy'
mask_file="/scr/ilz3/myelinconnect/new_groupavg/masks/fullmask_lh_rh_new.npy"
embed_file="/scr/ilz3/myelinconnect/new_groupavg/embed/indv/%s_sess1_both_smooth_3_embed.npy"
embed_dict_file="/scr/ilz3/myelinconnect/new_groupavg/embed/indv/%s_sess1_both_smooth_3_embed_dict.pkl"



mask = np.load(mask_file)

for sub in subjects:
    ts_files = []
    print sub
    print '...loading'
    for sess in sessions:
        rest_left = np.load(rest_file%(sub, 'lh', sess))
        rest_right = np.load(rest_file%(sub, 'rh', sess))
        ts_file = np.concatenate((rest_left, rest_right))
        ts_files.append(ts_file)
    
    print '...correlation'    
    upper_corr, full_shape = avg_correlation(ts_files)

    print '...embedding'
    embedding_recort, embedding_dict = embedding(upper_corr, full_shape, mask, n_embedding)

    print '...saving'
    np.save(embed_file%(sub),embedding_recort)
    pkl_out = open(embed_dict_file%(sub), 'wb')
    pickle.dump(embedding_dict, pkl_out)
    pkl_out.close()
    
    del upper_corr
    del embedding_recort
    del embedding_dict
