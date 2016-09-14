from __future__ import division
import numpy as np
import scipy as sp
import h5py
import hcp_corr
from vtk_rw import read_vtk, write_vtk
from clustering import embedding
import pickle
from mapalign import dist
import scipy.sparse as sparse


def chebapprox(profiles, degree):
    profiles=np.array(profiles)
    cheb_coeffs=np.zeros((profiles.shape[0],degree+1))
    cheb_polynoms=np.zeros((profiles.shape[0],profiles.shape[1]))
    for c in range(profiles.shape[0]):
        x=np.array(range(profiles.shape[1]))
        y=profiles[c]
        cheb_coeffs[c]=np.polynomial.chebyshev.chebfit(x, y, degree)
        cheb_polynoms[c]=np.polynomial.chebyshev.chebval(x, cheb_coeffs[c])
    return cheb_coeffs, cheb_polynoms


hemis=['rh']#,'rh']
smooth= 'smooth_3'
n_embedding = 10
affine_methods=['cauchy'] #['markov', 'cauchy']

compute_affinity = False
sparsify_affinity = False
embed_affinity = True



for hemi in hemis: 
    
    for affine_method in affine_methods:
        
        print hemi, affine_method
        
        embed_file="/scr/ilz3/myelinconnect/new_groupavg/embed/%s_%s_embed.npy"
        embed_dict_file="/scr/ilz3/myelinconnect/new_groupavg/embed/%s_%s_embed_dict.pkl"
        mask_file="/scr/ilz3/myelinconnect/new_groupavg/masks/%s_fullmask.npy"%hemi
        mesh_file="/scr/ilz3/myelinconnect/new_groupavg/surfs/lowres/%s_lowres_new.vtk"%hemi

        t1_file = '/scr/ilz3/myelinconnect/new_groupavg/profiles/%s/%s_group_avg_profiles_%s.npy'%(smooth, hemi, smooth)
       
        affinity_file = '/scr/ilz3/myelinconnect/new_groupavg/embed/profiles/%s_%s_%s_cheb_affinity.hdf5'%(hemi, smooth, affine_method)
        sparse_affinity_file = '/scr/ilz3/myelinconnect/new_groupavg/embed/profiles/%s_%s_%s_sparse_cheb_affinity_k%s.hdf5'
        embed_file = '/scr/ilz3/myelinconnect/new_groupavg/embed/profiles/%s_%s_%s_embed.npy'%(hemi, smooth, affine_method)
        embed_dict_file = '/scr/ilz3/myelinconnect/new_groupavg/embed/profiles/%s_%s_%s_embed_dict.pkl'%(hemi, smooth, affine_method)
 
        print 'loading data'
        v, f, d = read_vtk(mesh_file)
        t1=np.load(t1_file)
        full_shape=tuple((t1.shape[0], t1.shape[0]))

            
        if compute_affinity:
            print 'chebychev'
            t1_3_7 = t1[:,3:8]
            coeff, poly = chebapprox(t1_3_7, degree=4)
            #coeff = t1.copy()
            print 'affinity'
            t1_3_7_affine = dist.compute_affinity(coeff, method=affine_method)
            
            if sparsify_affinity:

                for neighbours in [50,100]:
                    print 'sparsify with %s neighbours'%str(neighbours)
                    sparse_affine = dist.compute_nearest_neighbor_graph(t1_3_7_affine, n_neighbors=neighbours)
                    components = sparse.csgraph.connected_components(sparse_affine, directed=False, connection='weak', return_labels=True)
                    if components[0] != 1:
                        print "sparse affinity matrix not fully connected with %s neighbours"%str(neighbours)
                        pass
                    else:
                        sparse_affine = sparse_affine.toarray()
                        #np.save(sparse_affinity_file%(hemi, smooth, affine_method, str(neighbours)), sparse_affine)
                        sparse_affine = sparse_affine[np.triu_indices_from(sparse_affine, k=1)]                
                        f = h5py.File(sparse_affinity_file%(hemi, smooth, affine_method, str(neighbours)), 'w')
                        f.create_dataset('upper', data=sparse_affine)
                        f.create_dataset('shape', data=full_shape)
                        f.close()
                        
            t1_3_7_affine = t1_3_7_affine[np.triu_indices_from(t1_3_7_affine, k=1)]                
            f = h5py.File(affinity_file, 'w')
            f.create_dataset('upper', data=t1_3_7_affine)
            f.create_dataset('shape', data=full_shape)
            f.close()
 
        if embed_affinity:
            
            if not compute_affinity:
                f = h5py.File(affinity_file, 'r')
                upper_affine = np.asarray(f['upper'])
                full_shape = tuple(f['shape'])
                f.close()
            else:
                upper_affine = t1_3_7_affine.copy()
                del t1_3_7_affine
                
            
            mask = np.load(mask_file)
            embedding_recort, embedding_dict = embedding(upper_affine, full_shape, mask, n_embedding)
        
            np.save(embed_file ,embedding_recort)
            pkl_out = open(embed_dict_file, 'wb')
            pickle.dump(embedding_dict, pkl_out)
            pkl_out.close()
        
