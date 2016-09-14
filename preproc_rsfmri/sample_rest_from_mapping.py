from __future__ import division
import numpy as np
from vtk_rw import read_vtk
import nibabel as nb
import pandas as pd
import os

'''
See project_mapping2struct.py for explanation
'''

subjects = pd.read_csv('/scr/ilz3/myelinconnect/subjects.csv')
subjects=list(subjects['DB'])
subjects.remove('KSMT')

hemis = ['rh', 'lh']
#sessions = ['1_1', '1_2', '2_1', '2_2']
sessions = ['2_2']

for hemi in hemis:
    for sess in sessions:
    
        mapping_x='/scr/ilz3/myelinconnect/mappings/rest2groupavg_surf/rest%s/x/%s_lowres_new_avgsurf_groupdata.vtk'%(sess, hemi)
        mapping_y='/scr/ilz3/myelinconnect/mappings/rest2groupavg_surf/rest%s/y/%s_lowres_new_avgsurf_groupdata.vtk'%(sess, hemi)
        mapping_z='/scr/ilz3/myelinconnect/mappings/rest2groupavg_surf/rest%s/z/%s_lowres_new_avgsurf_groupdata.vtk'%(sess, hemi)
        
        _, _, dx = read_vtk(mapping_x)
        _, _, dy = read_vtk(mapping_y)
        _, _, dz = read_vtk(mapping_z) 

        for sub in range(len(subjects)):

            rest_file='/scr/ilz3/myelinconnect/resting/final/%s_rest%s_denoised.nii.gz'%(subjects[sub], sess)
            out_file = '/scr/ilz3/myelinconnect/new_groupavg/rest/raw/%s_%s_rest%s.npy'%(subjects[sub], hemi, sess)
            
            rest_vol = nb.load(rest_file).get_data()
            rest_mesh = np.zeros((dx.shape[0], rest_vol.shape[-1]))
            
            coords = np.vstack((dx[:,sub],dy[:,sub],dz[:,sub]))
            coords = coords.T
            
            for vertex in range(rest_mesh.shape[0]):
                if np.all(coords[vertex]==0):
                    pass
                else:
                    coord = np.asarray(np.round(coords[vertex]), 'int64')
                    rest_mesh[vertex] = rest_vol[coord[0], coord[1], coord[2]]
            
            # save data
            np.save(out_file, rest_mesh)
            
            if os.path.isfile(out_file):
                print subjects[sub]+' '+hemi+' ' +sess+' finished'
