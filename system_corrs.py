import numpy as np
import pickle
import h5py
from mapalign import dist as mdist
from mapalign import embed as membed


corr_file = '/scr/ilz3/myelinconnect/new_groupavg/corr/rh_smooth_3_avg_corr.hdf5'
mask_file = '/scr/ilz3/myelinconnect/new_groupavg/masks/rh_smooth_3_motor.npy'


f = h5py.File(corr_file, 'r')
full_shape=tuple(f['shape'])
upper_corr=np.asarray(f['upper'])
f.close()
full_corr = np.zeros(tuple(full_shape))
full_corr[np.triu_indices_from(full_corr, k=1)] = np.nan_to_num(upper_corr)
full_corr += full_corr.T

source_nodes = np.load(mask_file)
region_corr = full_corr[source_nodes]

region_affinity = mdist.compute_affintiy(region_corr)
region_embed = membed.compute_diffusion_map(region_affinity, components=10)

corr_dict = {}
for node in source_nodes:
    corr_dict[node]=full_corr[node]
with open(corr_dict_file, 'w') as pkl_out:
    pickle.dump(corr_dict, pkl_out)
    
    
corr_array
corr_dict