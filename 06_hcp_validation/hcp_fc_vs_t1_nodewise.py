from __future__ import division
import numpy as np
import h5py
from scipy.spatial.distance import pdist
import scipy.stats as stats
from sklearn import linear_model
import nibabel as nb

'''
Calculate the node-wise correlation between FC and myelin
'''

fc_file = '/nobackup/kaiser2/dense_connectome/HCP_S900_820_rfMRI_MSMAll_groupPCA_d4500ROW_zcorr.dconn.nii'
myelin_file = '/nobackup/ilz3/myelinconnect/hcp/results/matrix_comparison/myelin_dist.hdf5'

fc_myelin_corr_file = '/nobackup/ilz3/myelinconnect/hcp/results/matrix_comparison/fc_myelin_corr.npy' 

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


print 'load myelin'
myelin, _ = load_full_mat(myelin_file, 0)

print 'load fc'
fc = np.tanh(np.squeeze(nb.load(fc_file).get_data()))
fc = fc[:myelin.shape[0], :myelin.shape[0]]


print 'node-wise correlation myelin and fc'
fc_myelin_corr = np.zeros((fc.shape[0],))
for i in range(fc.shape[0]):
    fc_myelin_corr[i] = stats.pearsonr(fc[i], myelin[i])[0]
np.save(fc_myelin_corr_file, fc_myelin_corr) 

