from __future__ import division
import numpy as np
import h5py
from scipy.spatial.distance import pdist

myelin_file = '/nobackup/ilz3/myelinconnect/hcp/data/myelin.npy'
mat_file = '/nobackup/ilz3/myelinconnect/hcp/results/matrix_comparison/myelin_dist.hdf5'

# load t1 data
print 'load data'
myelin = np.load(myelin_file)
size = myelin.shape[0]

# calculate upper triangle of myelin distances 
dist_upper = pdist(myelin[:,np.newaxis],'euclidean')

# save
print 'saving matrix'
f = h5py.File(mat_file, 'w')
f.create_dataset('upper', data=dist_upper, compression='gzip')
f.create_dataset('shape', data=(size, size))
f.close()