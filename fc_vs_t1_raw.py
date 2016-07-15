from __future__ import division
import numpy as np
import h5py
from scipy.spatial.distance import pdist
import scipy.stats as stats

fc_file = '/scr/ilz3/myelinconnect/new_groupavg/corr/both_smooth_3_avg_corr.hdf5'
t1_file = '/scr/ilz3/myelinconnect/new_groupavg/corr/both_smooth_avg_t1_dist.hdf5'
mask_file="/scr/ilz3/myelinconnect/new_groupavg/masks/fullmask_lh_rh_new.npy"
upper_mask_file="/scr/ilz3/myelinconnect/new_groupavg/masks/upper_fullmask_lh_rh_new.npy"
dist_file = '/scr/ilz3/myelinconnect/new_groupavg/corr/both_euclid_dist.hdf5'

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
#np.save(upper_mask_file, upper_mask)


print 'load t1'
f = h5py.File(t1_file, 'r')
t1 = np.asarray(f['upper'])
f.close()

print 'mask t1'
t1 = np.delete(t1, upper_mask)


print 'load dist'
f = h5py.File(dist_file, 'r')
dist = np.asarray(f['upper'])
f.close()

print 'mask dist'
dist = np.delete(dist, upper_mask)

print 'regress dist of t1'
#slope, intercept, r_value, p_value, std_err = stats.linregress(t1,dist)
#print slope, intercept, r_value, p_value, std_err

#resid_t1 = t1 - (intercept + slope*dist)

#del dist
#del t1

#print 'load fc'
#f = h5py.File(fc_file, 'r')
#fc = np.asarray(f['upper'])
#f.close()

#print 'mask fc'
#fc = np.delete(fc, upper_mask)

#print 'Pearson r', stats.pearsonr(fc, resid_t1)




#uncorrected df 7306284888
#Pearson r (-0.34304558035461219, 0.0)
#stats.pearsonr(fc, t1)[0]**2 : 0.11768027020083269