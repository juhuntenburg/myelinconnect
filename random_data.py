import numpy as np
from vtk_rw import read_vtk, write_vtk


iterations = 100

hemis = ['rh', 'lh']
mesh_file="/scr/ilz3/myelinconnect/new_groupavg/surfs/lowres/%s_lowres_new.vtk"
random_npy_file = '/scr/ilz3/myelinconnect/new_groupavg/model/random_data/raw/%s_random_normal_%s.npy'
random_vtk_file = '/scr/ilz3/myelinconnect/new_groupavg/model/random_data/raw/%s_random_normal_%s.vtk'


for hemi in hemis:
    print hemi
    
    v,f,d = read_vtk(mesh_file%hemi)
    data_shape = v.shape[0]
    
    for r in range(iterations):
        print r
        random_data = np.random.randn(data_shape)
        np.save(random_npy_file%(hemi, str(r)), random_data)
        write_vtk(random_vtk_file%(hemi, str(r)), v, f, data=random_data[:,np.newaxis])
        
