from __future__ import division
import numpy as np
import h5py
from scipy.spatial.distance import pdist
from vtk_rw import read_vtk

lh_mesh = '/scr/ilz3/myelinconnect/new_groupavg/surfs/lowres/lh_lowres_new.vtk'
rh_mesh = '/scr/ilz3/myelinconnect/new_groupavg/surfs/lowres/rh_lowres_new.vtk'
mat_file = '/scr/ilz3/myelinconnect/new_groupavg/corr/both_euclid_dist.hdf5'

# load t1 data
print 'load data'
v_left, _, _ = read_vtk(lh_mesh)
v_right, _, _ = read_vtk(rh_mesh)
v = np.concatenate((v_left, v_right))
size = v.shape[0]

# calculate t1 distances for each subject and average
print 'calc mat'
dist_upper = pdist(v,'euclidean')

# save
print 'saving matrix'
f = h5py.File(mat_file, 'w')
f.create_dataset('upper', data=dist_upper)
f.create_dataset('shape', data=(size, size))
f.close()