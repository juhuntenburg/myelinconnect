from __future__ import division
import numpy as np
import numexpr as ne
import pandas as pd
from correlations import avg_correlation
from clustering import embedding, kmeans, subcluster
from vtk_rw import read_vtk
import h5py
import pickle

ne.set_num_threads(ne.ncores-2)

subjects = pd.read_csv('/scr/ilz3/myelinconnect/subjects.csv')
subjects=list(subjects['DB'])
subjects.remove('KSMT')

subjects = ['BP4T']

smooths=['smooth_3'] #, 'raw', 'smooth_2']
hemis = ['rh'] #, 'lh']
sessions = ['1_1', '1_2' , '2_1', '2_2']
masktype = '025_5'
n_embedding = 10
#n_kmeans = range(2,21)
n_kmeans = [5,10,15]

calc_corr = False
calc_embed = False
calc_cluster = False
calc_subcluster = True

rest_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/rest/%s/%s_%s_rest%s_%s.npy'
#thr_corr_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/corr/%s_%s_thr%s_per_session_corr.hdf5'
corr_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/corr/%s_%s_avg_corr.hdf5'
embed_file="/scr/ilz3/myelinconnect/all_data_on_simple_surf/clust/%s/mask_%s/%s_embed_%s.npy"
embed_dict_file="/scr/ilz3/myelinconnect/all_data_on_simple_surf/clust/%s/mask_%s/%s_embed_%s_dict.pkl"
kmeans_file="/scr/ilz3/myelinconnect/all_data_on_simple_surf/clust/%s/mask_%s/%s_embed_%s_kmeans_%s.npy"
mask_file="/scr/ilz3/myelinconnect/all_data_on_simple_surf/masks/%s_fullmask_%s.npy"

mesh_file="//scr/ilz3/myelinconnect/all_data_on_simple_surf/surfs/lowres_%s_d.vtk"
subclust_file="/scr/ilz3/myelinconnect/all_data_on_simple_surf/clust/%s/mask_%s/%s_embed_%s_kmeans_%s_subclust.npy"

for smooth in smooths:
    print 'smooth '+smooth
    
    print 'mask '+masktype

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

            mask = np.load(mask_file%(hemi, masktype))
            embedding_recort, embedding_dict = embedding(upper_corr, full_shape, mask, n_embedding)

            np.save(embed_file%(smooth, masktype, hemi, str(n_embedding)),embedding_recort)
            pkl_out = open(embed_dict_file%(smooth, masktype, hemi, str(n_embedding)), 'wb')
            pickle.dump(embedding_dict, pkl_out)
            pkl_out.close()
            

        '''clustering'''
        if calc_cluster:
            if not calc_embed:
                embedding_recort = np.load(embed_file%(smooth, masktype, hemi, str(n_embedding)))
                mask = np.load(mask_file%(hemi, masktype))
            for nk in n_kmeans:
                print 'clustering %s'%str(nk)
                kmeans_recort = kmeans(embedding_recort, nk, mask)
                np.save(kmeans_file%(smooth, masktype, hemi, str(n_embedding), str(nk)),
                        kmeans_recort)
                
                if calc_subcluster:
                    print 'subclustering %s'%str(nk)
                    verts, faces, data = read_vtk(mesh_file%hemi)
                    subclust_arr=subcluster(kmeans_recort, faces)
                    np.save(subclust_file%(smooth, masktype, hemi, str(n_embedding), str(nk)), subclust_arr)   
                        

        '''subclustering'''
        if not calc_cluster:
            if calc_subcluster:
                
                verts, faces, data = read_vtk(mesh_file%hemi)
                
                for nk in n_kmeans:
                    print 'subclustering %s'%str(nk)
                    kmeans_recort = np.load(kmeans_file%(smooth, masktype, hemi, str(n_embedding), str(nk)))
                    subclust_arr=subcluster(kmeans_recort, faces)
                    np.save(subclust_file%(smooth, masktype, hemi, str(n_embedding), str(nk)), subclust_arr)  