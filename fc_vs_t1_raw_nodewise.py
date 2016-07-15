from __future__ import division
import numpy as np
import h5py
from scipy.spatial.distance import pdist
import scipy.stats as stats
from sklearn import linear_model

mask_file="/scr/ilz3/myelinconnect/new_groupavg/masks/fullmask_lh_rh_new.npy"
fc_file = '/scr/ilz3/myelinconnect/new_groupavg/corr/both_smooth_3_avg_corr.hdf5'
t1_file = '/scr/ilz3/myelinconnect/new_groupavg/corr/both_t1_dist.hdf5'
dist_file = '/scr/ilz3/myelinconnect/new_groupavg/corr/both_euclid_dist.hdf5'
resid_file = '/scr/ilz3/myelinconnect/new_groupavg/corr/both_residual_t1_after_dist.hdf5'
pure_corr_file = '/scr/ilz3/myelinconnect/new_groupavg/corr/fc_corr.npy'
dist_corr_file = '/scr/ilz3/myelinconnect/new_groupavg/corr/dist_corr.npy'
resid_corr_file = '/scr/ilz3/myelinconnect/new_groupavg/corr/resid_corr.npy'

print 'load_mask'
mask = np.load(mask_file)


print 'load t1'
f = h5py.File(t1_file, 'r')
t1 = np.asarray(f['upper'])
full_shape = tuple(f['shape'])
f.close()

print 'full matrix t1'
full_t1 = np.zeros(tuple(full_shape))
full_t1[np.triu_indices_from(full_t1, k=1)] = np.nan_to_num(t1)
del t1
full_t1 += full_t1.T
np.fill_diagonal(full_t1, 1)

print 'mask t1'
#full_t1 = np.delete(full_t1, mask, 0)
#full_t1 = np.delete(full_t1, mask, 1)

print 'load dist'
f = h5py.File(dist_file, 'r')
dist = np.asarray(f['upper'])
f.close()

print 'full matrix dist'
full_dist = np.zeros(tuple(full_shape))
full_dist[np.triu_indices_from(full_dist, k=1)] = np.nan_to_num(dist)
del dist
full_dist += full_dist.T
np.fill_diagonal(full_dist, 1)

print 'mask dist'
#full_dist = np.delete(full_dist, mask, 0)
#full_dist = np.delete(full_dist, mask, 1)


print 'node-wise dist regression'
dist_corr = np.zeros((full_t1.shape[0],))
dist_resid = np.zeros((full_t1.shape))


for i in range(full_t1.shape[0]):
    dist_corr[i] = stats.pearsonr(full_t1[i], full_dist[i])[0]
    
    clf = linear_model.LinearRegression()
    clf.fit(full_dist[i][:, np.newaxis], full_t1[i][:, np.newaxis])
    dist_resid[i] = full_t1[i] - np.squeeze(clf.predict(full_dist[i][:,np.newaxis]))
    
del full_dist

print 'saving dist corr'
np.save(dist_corr_file, dist_corr)
del dist_corr
    
#print 'saving residuals'
#f = h5py.File(resid_file, 'w')
#f.create_dataset('upper', data=dist_resid, chunks=(dist_resid.shape[0],), compression='gzip')
#f.create_dataset('shape', data=full_shape)
#f.close()


print 'load fc'
f = h5py.File(fc_file, 'r')
fc = np.asarray(f['upper'])
f.close()

print 'full matrix fc'
full_fc = np.zeros(tuple(full_shape))
full_fc[np.triu_indices_from(full_fc, k=1)] = np.nan_to_num(fc)
del fc
full_fc += full_fc.T
np.fill_diagonal(full_fc, 1)

print 'mask fc'
#full_fc = np.delete(full_fc, mask, 0)
#full_fc = np.delete(full_fc, mask, 1)
   
   
print 'corr fc with and without dist'
fc_corr = np.zeros((full_fc.shape[0],))
resid_corr = np.zeros((full_fc.shape[0],))

for i in range(full_fc.shape[0]):
    
    fc_corr[i] = stats.pearsonr(full_t1[i], full_fc[i])[0]
    resid_corr[i] = stats.pearsonr(dist_resid[i], full_fc[i])[0]
