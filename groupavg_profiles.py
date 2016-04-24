from vtk_rw import read_vtk, write_vtk
import numpy as np


smooth = 'smooth_6'
hemis = ['rh', 'lh']

vtk_file = '/scr/ilz3/myelinconnect/new_groupavg/profiles/%s/%s/%s_lowres_new_avgsurf_groupdata.vtk'
avg_npy_file = '/scr/ilz3/myelinconnect/new_groupavg/profiles/%s/%s_group_avg_profiles_%s.npy'
avg_vtk_file = '/scr/ilz3/myelinconnect/new_groupavg/profiles/%s/%s_group_avg_profiles_%s.vtk'

for hemi in hemis:
    
    avg_list = []
    
    for pro in range(11):
        
        v, f, d = read_vtk(vtk_file%(smooth, str(pro), hemi))
        avg_list.append(np.mean(d, axis=1))
        
        
    avg_array = np.asarray(avg_list).T
    np.save(avg_npy_file%(smooth, hemi, smooth), avg_array)
    write_vtk(avg_vtk_file%(smooth, hemi,smooth), v,f, data=avg_array)