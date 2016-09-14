from vtk_rw import read_vtk, write_vtk
import numpy as np


compartment = 'upper'
hemis = ['lh', 'rh']

vtk_file = '/scr/ilz3/myelinconnect/new_groupavg/profiles/%s/%s/%s_lowres_new_avgsurf_groupdata.vtk'
avg_npy_file = '/scr/ilz3/myelinconnect/new_groupavg/profiles/%s/%s_group_avg_profiles_%s.npy'
#avg_vtk_file = '/scr/ilz3/myelinconnect/new_groupavg/profiles/%s/%s_group_avg_profiles_%s.vtk'
both_hemi_npy_file = '/scr/ilz3/myelinconnect/new_groupavg/profiles/%s/both_group_avg_profiles_%s.npy'


both_hemis = []
for hemi in hemis:
    
    avg_list = []
    group_list = []
    
    for pro in range(11):
        
        v, f, d = read_vtk(vtk_file%(compartment, str(pro), hemi))
        avg_list.append(np.mean(d, axis=1))
        
    avg_array = np.asarray(avg_list).T
    both_hemis.append(avg_array)
    
    np.save(avg_npy_file%(compartment, hemi, compartment), avg_array)
    #write_vtk(avg_vtk_file%(compartment, hemi, compartment), v,f, data=avg_array)

#print 'ready'
both = np.concatenate(both_hemis)
np.save(both_hemi_npy_file%(compartment, compartment), both)