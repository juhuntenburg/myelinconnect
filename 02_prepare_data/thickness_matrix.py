from __future__ import division
import numpy as np
import h5py
from scipy.spatial.distance import pdist

thickness_file = '/nobackup/ilz3/myelinconnect/new_groupavg/thickness/%s_thickness_avg_mm.npy'
mat_file = '/nobackup/ilz3/myelinconnect/new_groupavg/corr/both_thickness_dist.hdf5'

# load t1 data
print 'load data'
thick_left = np.load(thickness_file%('lh'))
thick_right = np.load(thickness_file%('rh'))
thickness = np.concatenate((thick_left, thick_right))
size = thickness.shape[0]

# calculate t1 distances from average
dist_upper = pdist(thickness[:,np.newaxis],'euclidean')

# save
print 'saving matrix'
f = h5py.File(mat_file, 'w')
f.create_dataset('upper', data=dist_upper, compression='gzip')
f.create_dataset('shape', data=(size, size))
f.close()