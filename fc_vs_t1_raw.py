from __future__ import division
import numpy as np
import h5py
from scipy.spatial.distance import pdist
import scipy.stats as stats

fc_file = '/scr/ilz3/myelinconnect/new_groupavg/corr/both_smooth_3_avg_corr.hdf5'
t1_file = corr_file = '/scr/ilz3/myelinconnect/new_groupavg/corr/both_smooth_avg_t1_dist.hdf5'
mask_file="/scr/ilz3/myelinconnect/new_groupavg/masks/fullmask_lh_rh_new.npy"
upper_mask_file="/scr/ilz3/myelinconnect/new_groupavg/masks/upper_fullmask_lh_rh_new.npy"


print 'create mask'
f = h5py.File(fc_file, 'r')
full_shape = tuple(f['shape'])
f.close()

mask = np.load(mask_file)

fake_mat = np.zeros(full_shape)
fake_mat[mask] = 1
fake_mat[:,mask] = 1

upper_fake = fake_mat[np.triu_indices_from(fake_mat)]
del fake_mat
upper_mask=np.where(upper_fake)[0]
del upper_fake
np.save(upper_mask_file, upper_mask)



print 'load fc'
f = h5py.File(fc_file, 'r')
fc = np.asarray(f['upper'])
f.close()

print 'mask fc'
fc = np.delete(fc, upper_mask)

print 'load t1'
f = h5py.File(t1_file, 'r')
t1 = np.asarray(f['upper'])
f.close()

print 'mask t1'
t1 = np.delete(t1, upper_mask)

print 'uncorrected df', fc.shape[0]
print 'Pearson r', stats.pearsonr(fc, t1)




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