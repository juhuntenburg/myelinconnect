from __future__ import division
import numpy as np
import numexpr as ne
from clustering import embedding, kmeans
import h5py
import pickle

ne.set_num_threads(ne.ncores-2)
n_embedding = 10

corr_file = '/scr/ilz3/myelinconnect/new_groupavg/corr/rh_smooth_3_avg_corr.hdf5'
mask_file = "scr/ilz3/myelinconnect/new_groupavg/masks/rh_fullmask.npy"
embed_file="/scr/ilz3/myelinconnect/new_groupavg/embed/compartments/rh_smooth_motor_embed.npy"
embed_dict_file="/scr/ilz3/myelinconnect/new_groupavg/embed/compartments/rh_smooth_motor_embed_dict.pkl"


print '...load'
# load upper triangular avg correlation matrix
f = h5py.File(corr_file, 'r')
upper_corr = np.asarray(f['upper'])
full_shape = tuple(f['shape'])
f.close()

print '...embed'
mask = np.load(mask_file)
embedding_recort, embedding_dict = embedding(upper_corr, full_shape, mask, n_embedding)
np.save(embed_file%(comp),embedding_recort)
pkl_out = open(embed_dict_file%(comp), 'wb')
pickle.dump(embedding_dict, pkl_out)
pkl_out.close()