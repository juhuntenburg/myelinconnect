from __future__ import division
import numpy as np
import numexpr as ne
import pandas as pd
from correlations import avg_correlation
from clustering import t1embedding, kmeans
import h5py
import pickle

ne.set_num_threads(ne.ncores-2)

smooths=['smooth_3'] #, 'raw', 'smooth_2']
hemis = ['rh'] #, 'lh']
masktype = '02_4'
n_embedding = 10
layers = ['3_7']
#n_kmeans = range(2,21)

calc_euclid = False
calc_embed = True
calc_cluster = False

mask_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/masks/%s_fullmask_%s.npy'
t1_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/t1/avg_%s_profiles.npy'

euclid_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/t1/%s_euclidian_dist_%s.hdf5'
embed_file="/scr/ilz3/myelinconnect/all_data_on_simple_surf/clust/%s/mask_%s/t1_embed/%s_t1embed_%s_layer_%s.npy"
embed_dict_file="/scr/ilz3/myelinconnect/all_data_on_simple_surf/clust/%s/mask_%s/t1_embed/%s_t1embed_%s_layer_%s_dict.pkl"
#kmeans_file="/scr/ilz3/myelinconnect/all_data_on_simple_surf/clust/%s/mask_%s/%s_embed_%s_kmeans_%s.npy"


for smooth in smooths:
    print 'smooth '+smooth

    for hemi in hemis:
        print 'hemi '+hemi
        
        for layer in layers:

            '''euclidian distance'''
            if calc_euclid:
                print 'calculating euclidian distances'
                t1=np.load(t1_file%hemi)
                full_shape=tuple((t1.shape[0], t1.shape[0]))
                t1_diff = sp.spatial.distance.pdist(t1[:,int(layer[0]):(int(layer[-1])+1)], 'euclidean')
             
                print 'saving matrix'
                f = h5py.File(euclid_file%(hemi, layer), 'w')
                f.create_dataset('upper', data=t1_diff)
                f.create_dataset('shape', data=full_shape)
                f.close()
    
                if (not calc_embed and not calc_cluster):
                    del t1_diff
    
            '''embedding'''
            if calc_embed:
                print 'embedding'
    
                if not calc_euclid:
                    print '...load'
                    # load upper triangular avg correlation matrix
                    f = h5py.File(euclid_file%(hemi, layer), 'r')
                    t1_diff = np.asarray(f['upper'])
                    full_shape = tuple(f['shape'])
                    f.close()
    
                mask = np.load(mask_file%(hemi, masktype))
                
                t1_diff=1-(t1_diff/t1_diff.max())
                
                embedding_recort, embedding_dict = t1embedding(t1_diff, full_shape, mask, n_embedding)
    
                np.save(embed_file%(smooth, masktype, hemi, str(n_embedding)),embedding_recort, layer)
                pkl_out = open(embed_dict_file%(smooth, masktype, hemi, str(n_embedding), layer), 'wb')
                pickle.dump(embedding_dict, pkl_out)
                pkl_out.close()
                
    
#             '''clustering'''
#             if calc_cluster:
#                 if not calc_embed:
#                     embedding_recort = np.load(embed_file%(smooth, masktype, hemi, str(n_embedding), layer))
#                     mask = np.load(mask_file%(hemi, masktype))
#                 for nk in n_kmeans:
#                     print 'clustering %s'%str(nk)
#                     kmeans_recort = kmeans(embedding_recort, nk, mask)
#                     np.save(kmeans_file%(smooth, masktype, hemi, str(n_embedding), str(nk)),
#                             kmeans_recort)
