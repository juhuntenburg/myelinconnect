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

smooths=['smooth_3'] #, 'raw', 'smooth_2']
sessions = ['1_1', '1_2' , '2_1', '2_2']
n_embedding = 100

rest_file = '/scr/ilz3/myelinconnect/new_groupavg/rest/smooth_3/%s_%s_rest%s_smooth_3.npy'
mask_file="/scr/ilz3/myelinconnect/new_groupavg/masks/fullmask_lh_rh_new.npy"
corr_file = '/scr/ilz3/myelinconnect/new_groupavg/corr/both_smooth_3_avg_corr.hdf5'
embed_file="/scr/ilz3/myelinconnect/new_groupavg/embed/both_smooth_3_embed.npy"
embed_dict_file="/scr/ilz3/myelinconnect/new_groupavg/embed/both_smooth_3_embed_dict.pkl"

calc_corr = False
calc_embed = True


'''avg correlations'''
if calc_corr:
    ts_files = []
    for sub in subjects:
        for sess in sessions:
            rest_left = np.load(rest_file%(sub, 'lh', sess))
            rest_right = np.load(rest_file%(sub, 'rh', sess))
            ts_file = np.concatenate((rest_left, rest_right))
            ts_files.append(ts_file)

    print 'calculating average correlations'
    upper_corr, full_shape = avg_correlation(ts_files)

    print 'saving matrix'
    f = h5py.File(corr_file, 'w')
    f.create_dataset('upper', data=upper_corr)
    f.create_dataset('shape', data=full_shape)
    f.close()

    if (not calc_embed and not calc_cluster):
        del upper_corr

'''embedding'''
if calc_embed:
    print 'embedding'

    if not calc_corr:
        print '...load'
        # load upper triangular avg correlation matrix
        f = h5py.File(corr_file, 'r')
        upper_corr = np.asarray(f['upper'])
        full_shape = tuple(f['shape'])
        f.close()

    mask = np.load(mask_file)
    embedding_recort, embedding_dict = embedding(upper_corr, full_shape, mask, n_embedding)

    np.save(embed_file,embedding_recort)
    pkl_out = open(embed_dict_file, 'wb')
    pickle.dump(embedding_dict, pkl_out)
    pkl_out.close()
