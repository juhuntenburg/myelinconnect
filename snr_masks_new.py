import numpy as np
import pandas as pd
from simplification import sample_volume, sample_simple
from vtk_rw import read_vtk

subjects = pd.read_csv('/scr/ilz3/myelinconnect/subjects.csv')
subjects=list(subjects['DB'])
subjects.remove('KSMT')
hemis = ['rh', 'lh']
sessions = ['1_1', '1_2' , '2_1', '2_2']

tsnr_file = '/scr/ilz3/myelinconnect/new_groupavg/snr/tsnr/%s_%s_rest%s.npy'
inv2prob_file = '/scr/ilz3/myelinconnect/new_groupavg/snr/inv2prob/%s_lowres_new_trgsurf_groupdata.vtk'

tsnr_full_file = '/scr/ilz3/myelinconnect/new_groupavg/snr/tsnr/%s_tsnr_full.npy'
tsnr_min_file = '/scr/ilz3/myelinconnect/new_groupavg/snr/tsnr/%s_tsnr_min.npy'
inv2prob_full_file = '/scr/ilz3/myelinconnect/new_groupavg/snr/inv2prob/%s_inv2prob_full.npy'
inv2prob_min_file = '/scr/ilz3/myelinconnect/new_groupavg/snr/inv2prob/%s_inv2prob_min.npy'


for hemi in hemis:
    
    #n_vertices = np.load(tsnr_file%(subjects[0], hemi, sessions[0])).shape[0]
    #tsnr = np.zeros((n_vertices, len(subjects)*len(sessions)))
    
    _, _, inv2prob = read_vtk(inv2prob_file%(hemi))

#     tsnr_count = 0
#     for sub in range(len(subjects)): 
#         
#         for sess in sessions:
#              
#             print subjects[sub], sess
#              
#             tsnr_d = np.load(tsnr_file%(subjects[sub], hemi, sess))
#                  
#             # average across highres vertices that map to the same lowres vertex
#             tsnr[:,tsnr_count] = tsnr_d
#                          
#             tsnr_count += 1
            
    print 'saving'
    #save the minimum snr across all subjects and sessions
    #np.save(tsnr_full_file%(hemi), tsnr)
    #np.save(tsnr_min_file%(hemi), np.min(tsnr, axis=1))
    np.save(inv2prob_full_file%(hemi), inv2prob)
    np.save(inv2prob_min_file%(hemi), np.min(inv2prob, axis=1))