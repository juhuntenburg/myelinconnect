import numpy as np
from vtk_rw import read_vtk, write_vtk

iterations = 1000

hemis = ['rh', 'lh']
mesh_file="/scr/ilz3/myelinconnect/new_groupavg/surfs/lowres/%s_lowres_new.vtk"
t1_file = '/scr/ilz3/myelinconnect/new_groupavg/t1/smooth_3/%s_t1_avg_smooth_3.npy'
mask_file = '/scr/ilz3/myelinconnect/new_groupavg/masks/%s_fullmask_new.npy'
random_npy_file = '/scr/ilz3/myelinconnect/new_groupavg/model/random_data/raw/%s_random_normal_%s.npy'
random_vtk_file = '/scr/ilz3/myelinconnect/new_groupavg/model/random_data/raw/%s_random_normal_%s.vtk'


for hemi in hemis:
    print hemi
    
    v,f,d = read_vtk(mesh_file%hemi)
    
    for r in range(iterations):
        print r
        random_data = np.random.randn(v.shape[0])
        np.save(random_npy_file%(hemi, str(r)), random_data)
        write_vtk(random_vtk_file%(hemi, str(r)), v, f, data=random_data[:,np.newaxis])
    
    
    
#     mask = np.load(mask_file%hemi)
#     t1 = np.load(t1_file%hemi)
#      
#     # extend mask to nodes that have a t1avg < 1500
#     mask = np.unique(np.concatenate((mask,np.where(t1<=1500)[0])))
#     mask = np.asarray(mask, dtype='int64')
#     idcs=np.arange(0,v.shape[0])
#     nonmask=np.delete(idcs, mask)
#      
#     # mask t1 before permuting, so that only the values that are taken into 
#     # account for the modelling later are permuted
#     masked_t1 = np.delete(t1, mask)
#      
#     for r in range(iterations):
#         print r
#         # permute data in unmasked regions
#         # also fill masked regions with these random values 
#         # this way, the unmasked region has exactly the distribution liek the
#         # unmasked region in the t1, but there are no smoothing issues with the masked region
#         random_data = np.copy(t1)
#         random_data[nonmask] = np.random.permutation(masked_t1)
#         random_data[mask] = np.random.permutation(masked_t1)[:mask.shape[0]]
#          
#         np.save(random_npy_file%(hemi, str(r)), random_data)
#         write_vtk(random_vtk_file%(hemi, str(r)), v, f, data=random_data[:,np.newaxis])
        
