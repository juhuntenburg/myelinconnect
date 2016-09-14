from __future__ import division
import numpy as np
import h5py
from scipy.spatial.distance import pdist
import scipy.stats as stats

'''
Calculates correlations of upper triangle of T1 and FC matrices, repeat 
after (analytic) regression of euclidean distance.
'''

'''
------
INPUTS
------
'''

fc_file = '/scr/ilz3/myelinconnect/new_groupavg/corr/both_smooth_3_avg_corr.hdf5'
t1_file = '/scr/ilz3/myelinconnect/new_groupavg/corr/both_t1_dist.hdf5'
dist_file = '/scr/ilz3/myelinconnect/new_groupavg/corr/both_euclid_dist.hdf5'
mask_file="/scr/ilz3/myelinconnect/new_groupavg/masks/fullmask_lh_rh_new.npy"

'''
-------------
Load and mask
-------------
'''
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

print 'load and mask t1'
f = h5py.File(t1_file, 'r')
t1 = np.asarray(f['upper'])
f.close()
t1 = np.delete(t1, upper_mask)

print 'load and mask dist'
f = h5py.File(dist_file, 'r')
dist = np.asarray(f['upper'])
f.close()
dist = np.delete(dist, upper_mask)

print 'load and mask fc'
f = h5py.File(fc_file, 'r')
fc = np.asarray(f['upper'])
f.close()
fc = np.delete(fc, upper_mask)

'''
-------------------------------
Correlations before regression
-------------------------------
'''

t1fc_r = stats.pearsonr(fc, t1)[0]
print 'Pearson r fc vs t1', t1fc_r
#Pearson r fc vs t1 (-0.34304558035461219, 0.0)

t1dist_r = stats.pearsonr(dist, t1)[0]
print 'Pearson r dist vs t1', t1dist_r
#Pearson r dist vs t1 (0.0072327672598170925, 0.0)

fcdist_r = stats.pearsonr(dist, fc)[0]
print 'Pearson r dist vs fc', fcdist_r


'''
-------------------
Distance regression
-------------------
'''

print 'regress dist of t1'
t1_slope = t1dist_r * np.std(t1) / np.std(dist)
print '..slope', t1_slope
t1_intercept = np.mean(t1) - t1_slope * np.mean(dist)
print '..intercept', t1_intercept
t1resid = t1 - (t1_intercept + t1_slope * dist)
del t1

print 'regress dist of fc'
fc_slope = fcdist_r * np.std(fc) / np.std(dist)
print '..slope', fc_slope
fc_intercept = np.mean(fc) - fc_slope * np.mean(dist)
print '..intercept', fc_intercept
fcresid = fc - (fc_intercept + fc_slope * dist)
del fc


'''
----------------------------
Correlation after regression
----------------------------
'''
print 'Pearson r t1resd vs fcresid', stats.pearsonr(fcresid, t1resid)[0]
#Pearson r t1resd vs fcresid -0.372021220259