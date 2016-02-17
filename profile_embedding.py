from __future__ import division
import numpy as np
import scipy as sp
import h5py
import hcp_corr
from vtk_rw import read_vtk, write_vtk
from clustering import embedding
import pickle
from mapalign import dist


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


hemis=['lh','rh']
smooth= 'smooth_3'
n_embedding = 10
affine_methods=['markov', 'cauchy']

euclid = False
corr = False
affinity = True
embed_corr = False
embed_euclid = False
embed_affinity = True



for hemi in hemis: 
    
    for affine_method in affine_methods:
        
        print hemi, affine_method

        mesh_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/surfs/lowres_%s_d.vtk'%hemi
        mask_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/masks/%s_fullmask.npy'%hemi
        #t1_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/t1/%s/avg_%s_profiles_%s.npy'%(smooth, hemi, smooth)
        t1_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/t1/%s/avg_%s_coeffs_%s.npy'%(smooth, hemi, smooth)
        #euclid_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/t1/profile_embedding/%s_euclidian_dist_%s.hdf5'
        #corr_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/t1/profile_embedding/%s_profile_corr_%s.hdf5'
        affinity_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/embed/profiles/%s_%s_cheb_affinity_%s.hdf5'%(hemi, smooth, affine_method)
        embed_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/embed/profiles/%s_%s_embedding_%s_%s.npy'%(hemi, smooth, str(n_embedding), affine_method)
        embed_dict_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/embed/profiles/%s_%s_embedding_dict_%s_%s.pkl'%(hemi, smooth, str(n_embedding), affine_method)
        
        v, f, d = read_vtk(mesh_file)
        t1=np.load(t1_file)
        full_shape=tuple((t1.shape[0], t1.shape[0]))


        if euclid:
            print 'euclid'
            t1_3_7_diff = sp.spatial.distance.pdist(t1[:,3:8], 'euclidean')
            f = h5py.File(euclid_file%('rh', '3_7'), 'w')
            f.create_dataset('upper', data=t1_3_7_diff)
            f.create_dataset('shape', data=full_shape)
            f.close()
            del t1_3_7_diff
            
        if corr:
            print 'corr'
            t1_3_7_corr = hcp_corr.corrcoef_upper(t1[:,3:8])
            f = h5py.File(corr_file%('rh', '3_7'), 'w')
            f.create_dataset('upper', data=t1_3_7_corr)
            f.create_dataset('shape', data=full_shape)
            f.close()
            
        if affinity:
            #print 'chebychev'
            #t1_3_7 = t1[:,3:8]
            #coeff, poly = chebapprox(t1_3_7, degree=4)
            coeff = t1.copy()
            print 'affinity'
            t1_3_7_affine = dist.compute_affinity(coeff, method=affine_method)
            t1_3_7_affine = t1_3_7_affine[np.triu_indices_from(t1_3_7_affine, k=1)]
            f = h5py.File(affinity_file%(hemi, smooth), 'w')
            f.create_dataset('upper', data=t1_3_7_affine)
            f.create_dataset('shape', data=full_shape)
            f.close()
        
        if embed_corr:
            
            print 'embedding'
            
            if corr:
                upper_corr = t1_3_7_corr
            else:
                # load upper triangular avg correlation matrix
                f = h5py.File(corr_file%('rh', '3_7'), 'r')
                upper_corr = np.asarray(f['upper'])
                full_shape = tuple(f['shape'])
                f.close()
                
        
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
            
        
        if embed_affinity:
            
            if not affinity:
                f = h5py.File(affinity_file%('rh', '3_7'), 'r')
                upper_affine = np.asarray(f['upper'])
                full_shape = tuple(f['shape'])
                f.close()
            else:
                upper_affine = t1_3_7_affine.copy()
                del t1_3_7_affine
                
            
            mask = np.load(mask_file%(hemi))
            embedding_recort, embedding_dict = embedding(upper_affine, full_shape, mask, n_embedding)
        
            np.save(embed_file ,embedding_recort)
            pkl_out = open(embed_dict_file, 'wb')
            pickle.dump(embedding_dict, pkl_out)
            pkl_out.close()
        
