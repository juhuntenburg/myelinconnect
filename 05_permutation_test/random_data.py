import numpy as np
from vtk_rw import read_vtk, write_vtk

iterations = 1000

hemis = ['rh', 'lh']
mesh_file="/scr/ilz3/myelinconnect/new_groupavg/surfs/lowres/%s_lowres_new.vtk"
t1_file = '/scr/ilz3/myelinconnect/new_groupavg/t1/smooth_3/%s_t1_avg_smooth_3.npy'
mask_file = '/scr/ilz3/myelinconnect/new_groupavg/masks/%s_fullmask_new.npy'
random_npy_file = '/scr/ilz3/myelinconnect/new_groupavg/model/random_data/raw/%s_random_normal_%s.npy'
random_vtk_file = '/scr/ilz3/myelinconnect/new_groupavg/model/random_data/raw/%s_random_normal_%s.vtk'

'''
Create random datasets by drawing from a normal distribution and write them to 
group average surface for subsequent smoothing with cbstools.
'''


for hemi in hemis:
    print hemi
    
    v,f,d = read_vtk(mesh_file%hemi)
    
    for r in range(iterations):
        print r
        random_data = np.random.randn(v.shape[0])
        np.save(random_npy_file%(hemi, str(r)), random_data)
        write_vtk(random_vtk_file%(hemi, str(r)), v, f, data=random_data[:,np.newaxis])