import numpy as np
from vtk_rw import read_vtk

vtk_file = '/nobackup/ilz3/myelinconnect/new_groupavg/profiles/smooth_1.5/%s/%s_lowres_new_avgsurf_groupdata.vtk'
pro_file = '/nobackup/ilz3/myelinconnect/new_groupavg/profiles/smooth_1.5/%s_lowres_new_avgsurf_groupdata.npy'
pro_mean_file = '/nobackup/ilz3/myelinconnect/new_groupavg/profiles/smooth_1.5/%s_lowres_new_avgsurf_groupdata_mean.npy'

for hemi in ['lh', 'rh']:
    
    _, _, d = read_vtk(vtk_file%(0, hemi))
    pro = np.zeros((d.shape[0], d.shape[1], 11))
    
    for layer in range(11):
        
        _, _, d = read_vtk(vtk_file%(layer, hemi))
        pro[:,:,layer] = d
        
    pro_mean = np.mean(pro, axis=1)
    
    np.save(pro_file%(hemi), pro)
    np.save(pro_mean_file%(hemi), pro_mean)