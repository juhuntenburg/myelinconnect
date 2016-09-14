from __future__ import division
import numpy as np
import h5py
from scipy.spatial.distance import pdist
import scipy.stats as stats
from sklearn import linear_model

'''
Calculate the node-wise correlation between FC and T1 before and after
regression against euclidean distance, as well as node-wise correlation between
FC-distance and T1-distance.
'''

fc_file = '/scr/ilz3/myelinconnect/new_groupavg/corr/both_smooth_3_avg_corr.hdf5'
t1_file = '/scr/ilz3/myelinconnect/new_groupavg/corr/both_t1_dist.hdf5'
dist_file = '/scr/ilz3/myelinconnect/new_groupavg/corr/both_euclid_dist.hdf5'

fc_t1_corr_file = '/scr/ilz3/myelinconnect/new_groupavg/corr/fc_t1_corr.npy' #t1 vs fc
dist_t1_corr_file = '/scr/ilz3/myelinconnect/new_groupavg/corr/dist_t1_corr.npy' #t1 vs distance
fc_dist_corr_file = '/scr/ilz3/myelinconnect/new_groupavg/corr/fc_dist_corr.npy' #fc vs distance
double_resid_corr_file = '/scr/ilz3/myelinconnect/new_groupavg/corr/t1resid_fcresid_corr.npy' #fc after distance regression vs t1 after distance regression


def load_full_mat(file, fill_value):
    f = h5py.File(file, 'r')
    upper = np.asarray(f['upper'])
    full_shape = tuple(f['shape'])
    f.close()
    full = np.zeros(tuple(full_shape))
    full[np.triu_indices_from(full, k=1)] = np.nan_to_num(upper)
    del upper
    full += full.T
    np.fill_diagonal(full, fill_value)
    return full, full_shape

print 'load dist'
dist, full_shape = load_full_mat(dist_file, 0)

print 'load t1'
t1, _ = load_full_mat(t1_file, 0)

print 'load fc'
fc, _ = load_full_mat(fc_file, 1)

print 'node-wise correlation and regression t1 dist'
dist_t1_corr = np.zeros((t1.shape[0],))
dist_resid_t1 = np.zeros((t1.shape))
for i in range(t1.shape[0]):
    dist_t1_corr[i] = stats.pearsonr(t1[i], dist[i])[0]
    X = np.vstack((np.ones(dist[i].shape[0]), dist[i])).T
    model_dist = linear_model.LinearRegression()
    model_dist.fit(X, t1[i])
    dist_resid_t1[i] = t1[i] - model_dist.predict(X)
np.save(dist_t1_corr_file, dist_t1_corr)

print 'node-wise correlation and regression fc dist'
fc_dist_corr = np.zeros((t1.shape[0],))
dist_resid_fc = np.zeros((fc.shape))
for i in range(fc.shape[0]):
    fc_dist_corr[i] = stats.pearsonr(fc[i], dist[i])[0]
    X = np.vstack((np.ones(dist[i].shape[0]), dist[i])).T
    model_dist = linear_model.LinearRegression()
    model_dist.fit(X, fc[i])
    dist_resid_fc[i] = fc[i] - model_dist.predict(X)
np.save(fc_dist_corr_file, fc_dist_corr)

del dist

print 'node-wise correlation and regression (residual) t1 and fc'
fc_t1_corr = np.zeros((dist_resid_fc.shape[0],))
double_resid_corr = np.zeros((dist_resid_fc.shape[0],))
for i in range(dist_resid_fc.shape[0]):
    fc_t1_corr[i] = stats.pearsonr(fc[i], t1[i])[0]
    double_resid_corr[i] = stats.pearsonr(dist_resid_fc[i], dist_resid_t1[i])[0]
np.save(fc_t1_corr_file, fc_t1_corr) 
np.save(double_resid_corr_file, double_resid_corr)

