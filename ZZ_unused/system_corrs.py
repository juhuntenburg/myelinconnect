import numpy as np
import pickle
import h5py
from mapalign import dist as mdist
from mapalign import embed as membed


corr_file = '/scr/ilz3/myelinconnect/new_groupavg/corr/rh_smooth_3_avg_corr.hdf5'
#mask_file = '/scr/ilz3/myelinconnect/new_groupavg/masks/rh_smooth_3_motor.npy'
mask_file = '/scr/ilz3/myelinconnect/new_groupavg/masks/rh_unimodal.npy'
embed_file="/scr/ilz3/myelinconnect/new_groupavg/embed/rh_smooth_3_unimodal_embed.npy"
embed_dict_file="/scr/ilz3/myelinconnect/new_groupavg/embed/rh_smooth_3_unimodal_embed_dict.pkl"


f = h5py.File(corr_file, 'r')
full_shape=tuple(f['shape'])
upper_corr=np.asarray(f['upper'])
f.close()
full_corr = np.zeros(tuple(full_shape))
full_corr[np.triu_indices_from(full_corr, k=1)] = np.nan_to_num(upper_corr)
full_corr += full_corr.T

source_nodes = np.load(mask_file)
source_nodes = np.asarray(source_nodes, dtype='int64')
region_corr = full_corr[source_nodes]

region_affinity = mdist.compute_affinity(region_corr)
region_embed = membed.compute_diffusion_map(region_affinity, n_components=10)

np.save(embed_file, region_embed[0])
pkl_out = open(embed_dict_file, 'wb')
pickle.dump(region_embed[1], pkl_out)
pkl_out.close()
