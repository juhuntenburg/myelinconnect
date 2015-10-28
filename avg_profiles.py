from __future__ import division
import numpy as np
from vtk_rw import read_vtk, write_vtk
import pandas as pd
import pdb

subjects = pd.read_csv('/scr/ilz3/myelinconnect/subjects.csv')
subjects=list(subjects['DB'])
subjects.remove('KSMT')

hemis = ['rh']

layer_low = 3
layer_high = 8

pro_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/t1/%s_%s_profiles.npy'

mean_pro_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/t1/%s_%s_profiles_mean_3_7.npy'
avg_pro_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/t1/avg_%s_profiles.npy'
avg_mean_pro_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/t1/avg_%s_mean_3_7.npy'

avg_pro_surf = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/t1/avg_%s_profiles_%s.vtk'
avg_mean_surf = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/t1/avg_%s_mean_3_7.vtk'

surf_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/surfs/lowres_%s_d.vtk'

for hemi in hemis:
    
    get_size = np.load('/scr/ilz3/myelinconnect/all_data_on_simple_surf/t1/%s_%s_profiles.npy'%(subjects[0], hemi)).shape
    avg_pro = np.zeros((get_size[0], get_size[1]))
    avg_mean = np.zeros((get_size[0]))
    
    v,f,d = read_vtk(surf_file%(hemi))
    
    count = 0
    for sub in subjects:
        
        profiles = np.load(pro_file%(sub, hemi))
        
        
        try:
            mean_3_7 = np.load(mean_pro_file%(sub, hemi))
        
        except IOError:
            mean_3_7= np.mean(profiles[:,layer_low:layer_high], axis=1)
            np.save(mean_pro_file%(sub, hemi), mean_3_7)
        
        avg_pro += profiles
        avg_mean += mean_3_7
        count += 1
        
        print 'Finished '+sub
        
    avg_pro = avg_pro / count
    avg_mean = avg_mean / count
    np.save(avg_pro_file%(hemi), avg_pro)
    np.save(avg_mean_pro_file%(hemi), avg_mean)
    
    #pdb.set_trace()
    write_vtk(avg_mean_surf%(hemi), v, f, data=avg_mean[:,np.newaxis])
    
    for p in range(profiles.shape[1]):
        write_vtk(avg_pro_surf%(hemi, str(p)), v, f, data=avg_pro[:,p, np.newaxis])
    
    
    
        
        
        