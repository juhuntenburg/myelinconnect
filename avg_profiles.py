from __future__ import division
import numpy as np
from vtk_rw import read_vtk, write_vtk
import pandas as pd
import pdb

subjects = pd.read_csv('/scr/ilz3/myelinconnect/subjects.csv')
subjects=list(subjects['DB'])
subjects.remove('KSMT')

hemis = ['rh', 'lh']


pro_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/t1/smooth_3/%s_%s_profiles_smooth_3.npy'
#coeff_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/t1/smooth_3/%s_%s_coeff_smooth_3.npy'
avg_pro_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/t1/smooth_3/avg_%s_profiles_smooth_3.npy'
avg_r1_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/t1/smooth_3/avg_%s_r1_smooth_3.npy'
#avg_coeff_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/t1/smooth_3/avg_%s_coeffs_smooth_3.npy'

#avg_pro_surf = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/t1/avg_%s_profiles_%s.vtk'

surf_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/surfs/lowres_%s_d.vtk'

for hemi in hemis:
    
    get_size = np.load('/scr/ilz3/myelinconnect/all_data_on_simple_surf/t1/smooth_3/%s_%s_profiles_smooth_3.npy'%(subjects[0], hemi)).shape
    avg_pro = np.zeros((get_size[0], get_size[1]))
    avg_r1 = np.zeros((get_size[0], get_size[1]))
    #avg_coeff = np.zeros((get_size[0], 5))
    #avg_mean = np.zeros((get_size[0]))
    
    #v,f,d = read_vtk(surf_file%(hemi))
    
    count = 0
    for sub in subjects:
        
        profiles = np.load(pro_file%(sub, hemi))
        
        r_profiles=np.reciprocal(profiles)
        r_profiles[np.isinf(r_profiles)]=0
        
        #coeffs = np.load(coeff_file%(sub, hemi))
        
        #try:
        #    mean_3_7 = np.load(mean_pro_file%(sub, hemi))
        
        #except IOError:
        #    mean_3_7= np.mean(profiles[:,layer_low:layer_high], axis=1)
        #    np.save(mean_pro_file%(sub, hemi), mean_3_7)
        
        #avg_pro += profiles
        #avg_coeff += coeffs
        #avg_mean += mean_3_7
        avg_r1 += r_profiles
        count += 1
        
        print 'Finished '+sub
        
    #avg_pro = avg_pro / count
    #avg_coeff = avg_coeff / count
    #avg_mean = avg_mean / count
    avg_r1 = avg_r1 / count
    #np.save(avg_pro_file%(hemi), avg_pro)
    #np.save(avg_coeff_file%(hemi), avg_coeff)
    #np.save(avg_mean_pro_file%(hemi), avg_mean)
    np.save(avg_r1_file%(hemi), avg_r1)
    
    #pdb.set_trace()
    #write_vtk(avg_mean_surf%(hemi), v, f, data=avg_mean[:,np.newaxis])
    
    #for p in range(profiles.shape[1]):
    #    write_vtk(avg_pro_surf%(hemi, str(p)), v, f, data=avg_pro[:,p, np.newaxis])
    
    
    
        
        
        