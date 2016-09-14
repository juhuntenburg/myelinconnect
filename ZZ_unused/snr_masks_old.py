import numpy as np
import pandas as pd
from simplification import sample_volume, sample_simple
from vtk_rw import read_vtk

subjects = pd.read_csv('/scr/ilz3/myelinconnect/subjects.csv')
subjects=list(subjects['DB'])
subjects.remove('KSMT')
hemis = ['rh', 'lh']
sessions = ['1_1', '1_2' , '2_1', '2_2']

tsnr_file = '/scr/ilz3/myelinconnect/resting/preprocessed/%s/rest%s/realignment/corr_%s_rest%s_roi_tsnr.nii.gz'
label_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/labels/%s_%s_highres2lowres_labels.npy'
highres_func_file = '/scr/ilz3/myelinconnect/struct/surf_%s/orig2func/rest%s/%s_%s_mid_groupavgsurf.vtk'
inv2prob_highres_file = '/scr/ilz3/myelinconnect/struct/snr_mask/%s_%s_mid_tsnr.vtk'

tsnr_full_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/masks/%s_tsnr_full.npy'
tsnr_min_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/masks/%s_tsnr_min.npy'
inv2prob_full_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/masks/%s_inv2prob_full.npy'
inv2prob_min_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/masks/%s_inv2prob_min.npy'


for hemi in hemis:
    
    n_vertices = np.load(label_file%(subjects[0], hemi))[:,1].max()+1
    tsnr = np.zeros((n_vertices, len(subjects)*len(sessions)))
    inv2prob = np.zeros((n_vertices, len(subjects)))

    inv2prob_count = 0
    tsnr_count = 0
    for sub in subjects: 
        
        labels = np.load(label_file%(sub, hemi))[:,1]
        inv2prob_highres_v, inv2prob_highres_f, inv2prob_highres_d = read_vtk(inv2prob_highres_file%(sub, hemi))
        
        # average across highres vertices that map to the same lowres vertex
        inv2prob[:,inv2prob_count] = np.squeeze(sample_simple(inv2prob_highres_d, labels))
        
        inv2prob_count += 1
        
        for sess in sessions:
             
            print sub, sess
             
            highres_func_v, highres_func_f, highres_func_d = read_vtk(highres_func_file%(hemi, sess, sub, hemi))
              
            # sample resting state time series on highres mesh
            tsnr_highres = sample_volume(tsnr_file%(sub, sess, sub, sess), highres_func_v)
                 
            # average across highres vertices that map to the same lowres vertex
            tsnr[:,tsnr_count] = np.squeeze(sample_simple(tsnr_highres[:,np.newaxis], labels))
                         
            tsnr_count += 1
            
    print 'saving'
    #save the minimum snr across all subjects and sessions
    np.save(tsnr_full_file%(hemi), tsnr)
    np.save(tsnr_min_file%(hemi), np.min(tsnr, axis=1))
    np.save(inv2prob_full_file%(hemi), inv2prob)
    np.save(inv2prob_min_file%(hemi), np.min(inv2prob, axis=1))