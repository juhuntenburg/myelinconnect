from __future__ import division
import numpy as np
import scipy as sp
import h5py
import hcp_corr
from vtk_rw import read_vtk, write_vtk
from clustering import embedding
import pickle
from mapalign import dist

hemi='rh'
masktype='025_5'
n_embedding = 10

mesh_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/surfs/lowres_%s_d.vtk'%hemi
mask_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/masks/%s_fullmask_%s.npy'
t1_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/t1/avg_%s_profiles.npy'%(hemi)
euclid_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/t1/profile_embedding/%s_euclidian_dist_%s.hdf5'
corr_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/t1/profile_embedding/%s_profile_corr_%s.hdf5'
embed_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/t1/profile_embedding/%s_mask_%s_%s_%s.npy'
embed_dict_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/t1/profile_embedding/%s_mask_%s_%s_%s_dict.pkl'

v, f, d = read_vtk(mesh_file)
t1=np.load(t1_file)
full_shape=tuple((t1.shape[0], t1.shape[0]))


euclid = False
corr = False
embed_corr = False
embed_euclid = True

if euclid:
    print 'euclid'
    t1_3_7_diff = sp.spatial.distance.pdist(t1[:,3:8], 'euclidean')
    f = h5py.File(euclid_file%('rh', '3_7'), 'w')
    f.create_dataset('upper', data=t1_3_7_diff)
    f.create_dataset('shape', data=full_shape)
    f.close()
    del t1_3_7_diff
    
#     t1_2_8_diff = sp.spatial.distance.pdist(t1[:,2:9], 'euclidean')
#     f = h5py.File(euclid_file%('rh', '2_8'), 'w')
#     f.create_dataset('upper', data=t1_2_8_diff)
#     f.create_dataset('shape', data=full_shape)
#     f.close()
#     del t1_2_8_diff
    
    
if corr:
    print 'corr'
    t1_3_7_corr = hcp_corr.corrcoef_upper(t1[:,3:8])
    f = h5py.File(corr_file%('rh', '3_7'), 'w')
    f.create_dataset('upper', data=t1_3_7_corr)
    f.create_dataset('shape', data=full_shape)
    f.close()
    #del t1_3_7_corr


if embed_corr:
    
    print 'embedding'
    
    if not corr:
        # load upper triangular avg correlation matrix
        f = h5py.File(corr_file%('rh', '3_7'), 'r')
        upper_corr = np.asarray(f['upper'])
        full_shape = tuple(f['shape'])
        f.close()
    else:
        upper_corr = t1_3_7_corr

    mask = np.load(mask_file%(hemi, masktype))
    embedding_recort, embedding_dict = embedding(upper_corr, full_shape, mask, n_embedding)

    np.save(embed_file%(hemi, masktype, 'corr_embed', str(n_embedding)),embedding_recort)
    pkl_out = open(embed_dict_file%(hemi, masktype, 'corr_embed', str(n_embedding)), 'wb')
    pickle.dump(embedding_dict, pkl_out)
    pkl_out.close()
    


if embed_euclid:
    
    print 'loading'
    
    if not euclid:
        # load upper triangular avg correlation matrix
        f = h5py.File(euclid_file%('rh', '3_7'), 'r')
        upper_diff = np.asarray(f['upper'])
        full_shape = tuple(f['shape'])
        f.close()
    else:
        upper_diff = t1_3_7_diff
        
    full_diff = np.zeros(tuple(full_shape))
    full_diff[np.triu_indices_from(full_diff, k=1)] = np.nan_to_num(upper_diff)
    full_diff += full_diff.T
    
    print 'normalizing'
    full_diff_norm = dist.compute_affinity(full_diff)
    #del full_diff
    upper_diff_norm = full_diff_norm[np.triu_indices_from(full_diff_norm, k=1)]
    #del full_diff_norm
    
    print 'embedding'
    mask = np.load(mask_file%(hemi, masktype))
    embedding_recort, embedding_dict = embedding(upper_diff_norm, full_shape, mask, n_embedding)

    np.save(embed_file%(hemi, masktype, 'euclid_embed', str(n_embedding)),embedding_recort)
    pkl_out = open(embed_dict_file%(hemi, masktype, 'euclid_embed', str(n_embedding)), 'wb')
    pickle.dump(embedding_dict, pkl_out)
    pkl_out.close()
    



