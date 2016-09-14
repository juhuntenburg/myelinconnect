from __future__ import division
import numpy as np
from vtk_rw import read_vtk, write_vtk
import nibabel as nb
import pandas as pd
import os

'''
See project_mapping2struct.py for explanation
(here tsnr is sampled instead of the time series)
'''

subjects = pd.read_csv('/scr/ilz3/myelinconnect/subjects.csv')
subjects=list(subjects['DB'])
subjects.remove('KSMT')

hemis = ['rh', 'lh']
sessions = ['1_1', '1_2', '2_1', '2_2']

for hemi in hemis:
    for sess in sessions:
    
        mapping_x='/scr/ilz3/myelinconnect/mappings/rest2groupavg_surf/rest%s/x/%s_lowres_new_avgsurf_groupdata.vtk'%(sess, hemi)
        mapping_y='/scr/ilz3/myelinconnect/mappings/rest2groupavg_surf/rest%s/y/%s_lowres_new_avgsurf_groupdata.vtk'%(sess, hemi)
        mapping_z='/scr/ilz3/myelinconnect/mappings/rest2groupavg_surf/rest%s/z/%s_lowres_new_avgsurf_groupdata.vtk'%(sess, hemi)
        
        v, f, dx = read_vtk(mapping_x)
        _, _, dy = read_vtk(mapping_y)
        _, _, dz = read_vtk(mapping_z) 

        for sub in range(len(subjects)):

            tsnr_file='/scr/ilz3/myelinconnect/resting/preprocessed/%s/rest%s/realignment/corr_%s_rest%s_roi_tsnr.nii.gz'%(subjects[sub], sess, subjects[sub], sess)
            out_file = '/scr/ilz3/myelinconnect/new_groupavg/snr/tsnr/%s_%s_rest%s.npy'%(subjects[sub], hemi, sess)
            #out_vtk =  '/scr/ilz3/myelinconnect/new_groupavg/snr/tsnr/%s_%s_rest%s.vtk'%(subjects[sub], hemi, sess)
            
            tsnr_vol = nb.load(tsnr_file).get_data()
            tsnr_mesh = np.zeros(dx.shape[0],)
            
            coords = np.vstack((dx[:,sub],dy[:,sub],dz[:,sub]))
            coords = coords.T
            
            for vertex in range(tsnr_mesh.shape[0]):
                if np.all(coords[vertex]==0):
                    pass
                else:
                    coord = np.asarray(np.round(coords[vertex]), 'int64')
                    tsnr_mesh[vertex] = tsnr_vol[coord[0], coord[1], coord[2]]
            
            # save data
            np.save(out_file, tsnr_mesh)
            #write_vtk(out_vtk, v, f, data=np.squeeze(tsnr_mesh))
            
            if os.path.isfile(out_file):
                print subjects[sub]+' '+hemi+' ' +sess+' finished'
