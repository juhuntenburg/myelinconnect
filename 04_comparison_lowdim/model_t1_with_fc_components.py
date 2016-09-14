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
    clf.fit(masked_embed[:,maps], masked_t1)

    modelled_fit = clf.predict(masked_embed[:,maps])
    residuals = masked_t1 - clf.predict(masked_embed[:,maps])

    model_dict =dict()
    model_dict["maps"] = tuple(maps)
    model_dict["corr"] = stats.pearsonr(modelled_fit, masked_t1)[0]
    model_dict["rsquared"] = clf.score(masked_embed[:,maps], masked_t1)
    model_dict["bic"] = BIC(clf.coef_, residuals, masked_t1, sigma2_res)
    model_dict["res"] = (1./residuals.shape[0]) * np.sum(residuals**2)
    
    pkl_out = open(model_file, 'wb')
    pickle.dump(model_dict, pkl_out)
    pkl_out.close()     

    return


if __name__ == "__main__":
    
    ### load data
    print 'loading'
    rh_mesh_file='/scr/ilz3/myelinconnect/new_groupavg/surfs/lowres/rh_lowres_new.vtk'
    lh_mesh_file='/scr/ilz3/myelinconnect/new_groupavg/surfs/lowres/lh_lowres_new.vtk'
    full_mask_file='/scr/ilz3/myelinconnect/new_groupavg/masks/fullmask_lh_rh_new.npy'
    rh_t1_file='/scr/ilz3/myelinconnect/new_groupavg/t1/smooth_1.5/rh_t1_avg_smooth_1.5.npy'
    lh_t1_file='/scr/ilz3/myelinconnect/new_groupavg/t1/smooth_1.5/lh_t1_avg_smooth_1.5.npy'
    embed_file='/scr/ilz3/myelinconnect/new_groupavg/embed/both_smooth_3_embed.npy'
    embed_dict_file='/scr/ilz3/myelinconnect/new_groupavg/embed/both_smooth_3_embed_dict.pkl'
    model_file_template = '/scr/ilz3/myelinconnect/new_groupavg/model/linear_combination/t1avg/smooth_1.5/model_comparison/model_%s.pkl'
    
    lv,lf,_ = read_vtk(lh_mesh_file)
    lh_t1 = np.load(lh_t1_file)
    rv,rf,_ = read_vtk(rh_mesh_file)
    rh_t1 = np.load(rh_t1_file)
    
    mask = np.load(full_mask_file)
    
    # prepare embedding (normalized from entry in dict)
    pkl_in = open(embed_dict_file, 'r')
    embed_dict=pickle.load(pkl_in)
    pkl_in.close()
    
    embed_masked = np.zeros((embed_dict['vectors'].shape[0], embed_dict['vectors'].shape[1]-1))
    for comp in range(100):
        embed_masked[:,comp]=(embed_dict['vectors'][:,comp+1]/embed_dict['vectors'][:,0])
    
    # unmask the embedding, that has been saved in masked form
    idcs=np.arange(0,(lv.shape[0]+rv.shape[0]))
    nonmask=np.delete(idcs, mask)
    embed = np.zeros(((lv.shape[0]+rv.shape[0]),100))
    embed[nonmask] = embed_masked
    
    # extend mask to nodes that have a t1avg < 1500
    full_t1 = np.concatenate((lh_t1, rh_t1))
    fullmask = np.unique(np.concatenate((mask,np.where(full_t1<=1500)[0])))
    fullmask = np.asarray(fullmask, dtype='int64')
    nonmask_bigmask=np.delete(idcs, fullmask)
    
    # mask embedding and t1
    masked_t1 = np.delete(full_t1, fullmask)
    masked_embed = np.delete(embed, fullmask, axis=0)
    
    # estimate sigma2_res from noise in T1
    Gl = graph_from_mesh(lv, lf)
    Gr = graph_from_mesh(rv, rf)
    
    left_medians= []
    for li in range(lv.shape[0]):
        neigh = Gl.neighbors(li)
        neigh_t1 = [full_t1[n] for n in neigh if not full_t1[n]==0]
        neigh_dist = [np.abs(full_t1[li] - nt) for nt in neigh_t1]
        left_medians.append(np.median(neigh_dist))
        
        
    right_medians = []
    for ri in range(rv.shape[0]):
        neigh = [x + lv.shape[0] for x in Gr.neighbors(ri)]
        neigh_t1 = [full_t1[n] for n in neigh if not full_t1[n]==0]
        neigh_dist = [np.abs(full_t1[ri+lv.shape[0]] - nt) for nt in neigh_t1]
        right_medians.append(np.median(neigh_dist))
        
        
    all_medians = np.concatenate((left_medians, right_medians))
    masked_medians = np.delete(all_medians, fullmask)
    noise_median = 0.5*np.nanmedian(masked_medians)/0.67448975
    sigma2_res = noise_median**2
    
    # list all possible combinations of maps
    maps = range(20)
    combinations = []
    for i in range(len(maps)):
        element = [list(x) for x in itertools.combinations(maps, i+1)]
        combinations.extend(element)


    Parallel(n_jobs=30)(delayed(fit_model)(i)
                        for i in range(len(combinations)))
