from __future__ import division
import numpy as np
from vtk_rw import read_vtk
from sklearn import linear_model
import scipy.stats as stats
import pickle
import itertools
import pandas as pd
from graphs import graph_from_mesh
import gdist
from joblib import Parallel, delayed
from hcp_utils import load_hcp

'''
Fit all possible combinations of the first 20 FC components to T1 and assess
model fit using BIC.
'''

## function to calculate BIC
def BIC(params, residuals, data, sigma2_res):
    p = params.shape[0]
    n = residuals.shape[0]
    data_range = data.max()-data.min()
    bic = (1-p) * np.log(2*np.pi*sigma2_res) + (1./sigma2_res) * (1./n) * np.sum(residuals**2) + p*np.log(data_range**2)
    return bic


def fit_model(c):
    
    print c
    maps=combinations[c]
    model_file = model_file_template%str(c)
    clf = linear_model.LinearRegression()
    clf.fit(masked_embed[:,maps], masked_myelin)

    modelled_fit = clf.predict(masked_embed[:,maps])
    residuals = masked_myelin - clf.predict(masked_embed[:,maps])

    model_dict =dict()
    model_dict["maps"] = tuple(maps)
    model_dict["corr"] = stats.pearsonr(modelled_fit, masked_myelin)[0]
    model_dict["rsquared"] = clf.score(masked_embed[:,maps], masked_myelin)
    model_dict["bic"] = BIC(clf.coef_, residuals, masked_myelin, sigma2_res)
    model_dict["res"] = (1./residuals.shape[0]) * np.sum(residuals**2)
    
    pkl_out = open(model_file, 'wb')
    pickle.dump(model_dict, pkl_out)
    pkl_out.close()     

    return


if __name__ == "__main__":
    
    ### load data
    print 'loading'
    data_dir = '/nobackup/ilz3/myelinconnect/hcp/'
    myelin_file= data_dir+'data/myelin.npy'
    embed_file=data_dir+'data/gradients.npy'
    model_file_template = data_dir+'results/model_comparison/model_%s.pkl'
    
    lsurf, rsurf, lidx, ridx = load_hcp(data_dir+'surfaces/')
    lv, lf = lsurf[0], lsurf[1]
    rv, rf = rsurf[0], rsurf[1]
    cortex = np.concatenate((lidx, ridx+lv.shape[0]))
    
    masked_myelin = np.load(myelin_file)
    masked_embed = np.load(embed_file)
    
    # unmask embedding and myelin
    full_embed = np.zeros(((lv.shape[0]+rv.shape[0]),300))
    full_embed[cortex] = masked_embed
    
    full_myelin = np.zeros(lv.shape[0]+rv.shape[0],)
    full_myelin[cortex] = masked_myelin
    
    
    # estimate sigma2_res from noise in T1
    Gl = graph_from_mesh(lv, lf)
    Gr = graph_from_mesh(rv, rf)
    
    left_medians= []
    for li in range(lv.shape[0]):
        neigh = Gl.neighbors(li)
        neigh_myelin = [full_myelin[n] for n in neigh if not full_myelin[n]==0]
        neigh_dist = [np.abs(full_myelin[li] - nt) for nt in neigh_myelin]
        left_medians.append(np.median(neigh_dist))
        
    right_medians = []
    for ri in range(rv.shape[0]):
        neigh = [x + lv.shape[0] for x in Gr.neighbors(ri)]
        neigh_myelin = [full_myelin[n] for n in neigh if not full_myelin[n]==0]
        neigh_dist = [np.abs(full_myelin[ri+lv.shape[0]] - nt) for nt in neigh_myelin]
        right_medians.append(np.median(neigh_dist))
        
    all_medians = np.concatenate((left_medians, right_medians))
    masked_medians = all_medians[cortex]
    noise_median = 0.5*np.nanmedian(masked_medians)/0.67448975
    sigma2_res = noise_median**2
    print sigma2_res
    
    # list all possible combinations of maps
    maps = range(20)
    combinations = []
    for i in range(len(maps)):
        element = [list(x) for x in itertools.combinations(maps, i+1)]
        combinations.extend(element)


    Parallel(n_jobs=30)(delayed(fit_model)(i)
                        for i in range(len(combinations)))
