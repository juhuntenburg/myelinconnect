from __future__ import division
import numpy as np
import nibabel as nb
import h5py
from scipy.spatial.distance import pdist
import scipy.stats as stats

'''
Calculates correlations of upper triangle of myelin and FC matrices
'''

'''
------
INPUTS
------
'''

fc_file = '/nobackup/kaiser2/dense_connectome/HCP_S900_820_rfMRI_MSMAll_groupPCA_d4500ROW_zcorr.dconn.nii'
myelin_file = '/nobackup/ilz3/myelinconnect/hcp/results/matrix_comparison/myelin_dist.hdf5'

'''
------
Load 
------
'''

print 'load myelin'
f = h5py.File(myelin_file, 'r')
myelin_upper = np.asarray(f['upper'])
size = tuple(f['shape'])[0]
f.close()

print 'load fc'
fc = np.tanh(np.squeeze(nb.load(fc_file).get_data()))
fc=fc[:size, :size]
fc_upper = fc[np.triu_indices_from(fc,1)]
del fc

'''
-------------
Correlations 
------------
'''

myelinfc = stats.pearsonr(fc_upper, myelin_upper)
print 'Pearson r fc vs t1', myelinfc
#(-0.10930471896558638, 0.0)

myelinfc_spear = stats.pearsonr(fc_upper, myelin_upper)
print 'Spearman r fc vs t1', myelinfc_spear
#SpearmanrResult(correlation=-0.095089260601349215, pvalue=0.0)
