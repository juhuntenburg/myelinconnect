from __future__ import division
import numpy as np
import numexpr as ne
import pandas as pd
from correlations import avg_correlation
from clustering import embedding, kmeans
import h5py

ne.set_num_threads(ne.ncores-2)

subjects = pd.read_csv('/scr/ilz3/myelinconnect/subjects.csv')
subjects=list(subjects['DB'])
subjects.remove('KSMT')

smooths=['embed_smooth_2', 'embed_smooth_3'] #, 'raw', 'smooth_2']
hemis = ['rh'] #, 'lh']
sessions = ['1_1', '1_2' , '2_1', '2_2']
n_embedding = 10
n_kmeans = range(2,21)

calc_corr = False
calc_embed = False
calc_cluster = True

rest_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/rest/%s/%s_%s_rest%s_%s.npy'
#thr_corr_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/corr/%s_%s_thr%s_per_session_corr.hdf5'
corr_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/corr/%s_%s_avg_corr.hdf5'
embed_file="/scr/ilz3/myelinconnect/all_data_on_simple_surf/clust/%s/%s_embed_%s.npy"
kmeans_file="/scr/ilz3/myelinconnect/all_data_on_simple_surf/clust/%s/%s_embed_%s_kmeans_%s.npy"
mask_file="/scr/ilz3/myelinconnect/all_data_on_simple_surf/masks/%s_mask.1D.roi"

for smooth in smooths:
    print 'smooth '+smooth

    for hemi in hemis:
        print 'hemi '+hemi

        '''avg correlations'''
        if calc_corr:
            ts_files = []
            for sub in subjects:
                for sess in sessions:
                    ts_files.append(rest_file%(smooth, sub, hemi, sess, smooth))

            print 'calculating average correlations'
            upper_corr, full_shape = avg_correlation(ts_files)

            print 'saving matrix'
            f = h5py.File(corr_file%(hemi, smooth), 'w')
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
                f = h5py.File(corr_file%(hemi, smooth), 'r')
                upper_corr = np.asarray(f['upper'])
                full_shape = tuple(f['shape'])
                f.close()

            mask = np.loadtxt(mask_file%(hemi),dtype='int64')[:,0]
            embedding_recort = embedding(upper_corr, full_shape, mask, n_embedding)

            np.save(embed_file%(smooth, hemi, str(n_embedding)),embedding_recort)

        '''clustering'''
        if calc_cluster:
            if not calc_embed:
                embedding_recort = np.load(embed_file%(smooth, hemi, str(n_embedding)))
                mask = np.loadtxt(mask_file%(hemi),dtype='int64')[:,0]
            for nk in n_kmeans:
                print 'clustering %s'%str(nk)
                kmeans_recort = kmeans(embedding_recort, nk, mask)
                np.save(kmeans_file%(smooth, hemi, str(n_embedding), str(nk)),
                        kmeans_recort)
