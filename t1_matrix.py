from __future__ import division
import numpy as np
import h5py
from scipy.spatial.distance import pdist

t1_file = '/scr/ilz3/myelinconnect/new_groupavg/t1/smooth_1.5/%s_t1_groupdata_smooth_1.5.npy'
mat_file = corr_file = '/scr/ilz3/myelinconnect/new_groupavg/corr/both_smooth_avg_t1_dist.hdf5'

# load t1 data
print 'load data'
t1_left = np.load(t1_file%('lh'))
t1_right = np.load(t1_file%('rh'))
t1 = np.concatenate((t1_left, t1_right))
size = t1[:,0].shape[0]


# calculate t1 distances for each subject and average
dist_upper =  np.zeros((size**2-size)/2)

for i in range(t1.shape[1]):
    print 'calc sub %i'%i
    dist_upper += pdist(t1[:,i,np.newaxis],'euclidean')
    
print 'divide'
dist_upper /= t1.shape[1]


# save
print 'saving matrix'
f = h5py.File(mat_file, 'w')
f.create_dataset('upper', data=dist_upper)
f.create_dataset('shape', data=(size, size))
f.close()