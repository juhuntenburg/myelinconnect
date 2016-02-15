from __future__ import division
import numpy as np
from vtk_rw import read_vtk, write_vtk
import pandas as pd


subjects = pd.read_csv('/scr/ilz3/myelinconnect/subjects.csv')
subjects=list(subjects['DB'])
subjects.remove('KSMT')

hemis = ['rh', 'lh']
sessions = ['1_1', '1_2', '2_1', '2_2']

mesh_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/surfs/lowres_%s_d.vtk'
data_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/t1/smooth_3/%s_%s_coeff_smooth_3.npy'
vtk_data_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/t1/smooth_3/%s_%s_coeff_raw_smoothdata.vtk'

#data_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/t1/raw/%s_%s_profiles.npy'
#vtk_data_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/t1/raw/%s_%s_profiles_raw.vtk'

mode = 'vtk2npy'

if mode == 'npy2vtk' :
    
    print 'npy to vtk'
    
    for hemi in hemis: 
        
        v,f,d = read_vtk(mesh_file%(hemi))
        
        for sub in subjects: 
            
#            print hemi, sub
#            data = np.load(data_file%(sub, hemi))
#            write_vtk(vtk_data_file%(sub, hemi), v,f,data=data)
            
             for sess in sessions:
                 
                 print hemi, sub, sess
             
                 data = np.load(data_file%(sub, hemi, sess))
                 write_vtk(vtk_data_file%(sub, hemi, sess), v,f,data=data)

                
elif mode == 'vtk2npy':
    
    print 'vtk to npy'
    
    for hemi in hemis: 
        
        for sub in subjects: 
            
                print hemi, sub
            
                v,f,d = read_vtk(vtk_data_file%(sub, hemi))
                np.save(data_file%(sub, hemi), d)
            
#             for sess in sessions:
#                 
#                 print hemi, sub, sess
#             
#                 v,f,d = read_vtk(vtk_data_file%(sub, hemi, sess))
#                 np.save(data_file%(sub, hemi, sess), d)
            


