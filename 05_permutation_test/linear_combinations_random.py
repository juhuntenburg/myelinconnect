import numpy as np
from vtk_rw import read_vtk
from sklearn import linear_model
import scipy.stats as stats
import pickle
from histogram_matching import hist_match
from utils import tupler
from joblib import Parallel, delayed

'''
Try to fit random data with linear combinations of embedding components.
Perform histogram matching of random data to original T1 data first.
'''

def fit_iteration((smooth, iteration)):
    
    print smooth, iteration, hemi

    _,_,t1 = read_vtk(random_file%(smooth, hemi, str(iteration)))
    masked_t1 = np.delete(t1, mask)
    masked_orig = np.delete(t1_orig, mask)
    
    # histogramm matching to original t1
    masked_t1 = hist_match(masked_t1, masked_orig)
    
    clf = linear_model.LinearRegression()
    clf.fit(masked_embed[:,maps], masked_t1)
    
    modelled_fit = clf.predict(masked_embed[:,maps])
    residuals = masked_t1 - clf.predict(masked_embed[:,maps])
    
    # write data back into cortex dimensions to avoid confusion
    modelled_fit_cortex = np.zeros((t1.shape[0])) 
    modelled_fit_cortex[nonmask_small]=modelled_fit
    residuals_cortex = np.zeros((t1.shape[0]))  
    residuals_cortex[nonmask_small]=residuals
    t1_cortex = np.zeros((t1.shape[0]))
    t1_cortex[nonmask_small]=masked_t1
    
    model_dict = dict()
    model_dict['maps']=maps
    model_dict['coef']=clf.coef_
    model_dict['modelled_fit']=modelled_fit_cortex
    model_dict['residuals']=residuals_cortex
    model_dict['t1']=t1_cortex
    model_dict['score'] = clf.score(masked_embed[:,maps], masked_t1)
    model_dict['corr'] = stats.pearsonr(modelled_fit, masked_t1)
    
    pkl_out = open(model_file%(smooth, hemi, iteration, maps_str), 'wb')
    pickle.dump(model_dict, pkl_out)
    pkl_out.close()
    
    return


if __name__ == "__main__":

    #mesh_file = '/scr/ilz3/myelinconnect/new_groupavg/surfs/lowres/%s_lowres_new.vtk'
    t1_file = '/scr/ilz3/myelinconnect/new_groupavg/t1/smooth_3/%s_t1_avg_smooth_3.npy'
    mask_file="/scr/ilz3/myelinconnect/new_groupavg/masks/%s_fullmask_new.npy"
    fullmask_file = '/scr/ilz3/myelinconnect/new_groupavg/masks/fullmask_lh_rh_new.npy'
    random_file = '/scr/ilz3/myelinconnect/new_groupavg/model/random_data/%s/%s_random_normal_%s_smoothdata.vtk'
    embed_dict_file="/scr/ilz3/myelinconnect/new_groupavg/embed/both_smooth_3_embed_dict.pkl"
    model_file = '/scr/ilz3/myelinconnect/new_groupavg/model/linear_combination/random/%s/%s_random_%s_t1avg_by_fc_maps_%s.pkl'
    
    
    random_smooths=['smooth_1.5', 'smooth_3', 'smooth_6', 'smooth_12', 'smooth_24']
    iterations = range(1000)
    maps =  range(20)
    maps_str = '0' #, 'best'
    hemi = 'rh' #'lh'
    
    # load t1 file for dimensions and histogramm
    t1_left = np.load(t1_file%('lh'))
    t1_right = np.load(t1_file%('rh'))
    
    # load embedding dict
    pkl_in = open(embed_dict_file, 'r')
    embed_dict=pickle.load(pkl_in)
    pkl_in.close()
    
    # prepare embedding (normalized from entry in dict)
    embed_masked = np.zeros((embed_dict['vectors'].shape[0], embed_dict['vectors'].shape[1]-1))
    for comp in range(100):
        embed_masked[:,comp]=(embed_dict['vectors'][:,comp+1]/embed_dict['vectors'][:,0])
    
    # unmask embed which has been saved in masked form
    fullmask = np.load(fullmask_file)
    idcs_full=np.arange(0,(t1_left.shape[0]+t1_right.shape[0]))
    nonmask_full=np.delete(idcs_full, fullmask)
    embed_full = np.zeros(((t1_left.shape[0]+t1_right.shape[0]),100))
    embed_full[nonmask_full] = embed_masked
    
    # according to hemisphere, slice embedding vectors
    # create indices and copy correct t1
    if hemi == 'lh':
        embed = embed_full[:t1_left.shape[0]]
        idcs_small = np.arange(0,t1_left.shape[0])
        t1_orig = np.copy(t1_left)
    elif hemi =='rh':
        embed = embed_full[t1_left.shape[0]:]
        idcs_small = np.arange(0,t1_right.shape[0])
        t1_orig = np.copy(t1_right)
        
    #extend mask to regions < 1500
    mask = np.load(mask_file%hemi)
    mask = np.unique(np.concatenate((mask, np.where(t1_orig<=1500)[0])))
    nonmask_small = np.delete(idcs_small, mask)
    
    # mask embedding
    masked_embed = np.delete(embed, mask, axis=0)
    
    # run model fit for all smooths and iterations in parallel
    Parallel(n_jobs=50)(delayed(fit_iteration)(i) 
                       for i in tupler(random_smooths, iterations))
          