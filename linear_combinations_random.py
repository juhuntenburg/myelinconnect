import numpy as np
from vtk_rw import read_vtk
from sklearn import linear_model
import scipy.stats as stats
import pickle



mesh_file = '/scr/ilz3/myelinconnect/new_groupavg/surfs/lowres/%s_lowres_new.vtk'
mask_file="/scr/ilz3/myelinconnect/new_groupavg/masks/fullmask_lh_rh_new.npy"
random_file = '/scr/ilz3/myelinconnect/new_groupavg/model/random_data/%s/%s_random_normal_%s_smoothdata.vtk'
embed_dict_file="/scr/ilz3/myelinconnect/new_groupavg/embed/both_smooth_3_embed_dict.pkl"
model_file = '/scr/ilz3/myelinconnect/new_groupavg/model/linear_combination/random/random_%s_%s_t1avg_by_fc_maps_%s.pkl'


all_maps = [[0],range(10)]
all_maps_str = ['0', '0_to_10']
random_smooths=['smooth_3', 'smooth_6', 'smooth_12', 'smooth_20']

# prepare embedding (normalized from entry in dict)
pkl_in = open(embed_dict_file, 'r')
embed_dict=pickle.load(pkl_in)
pkl_in.close()

embed_masked = np.zeros((embed_dict['vectors'].shape[0], embed_dict['vectors'].shape[1]-1))
for comp in range(100):
    embed_masked[:,comp]=(embed_dict['vectors'][:,comp+1]/embed_dict['vectors'][:,0])

mask = np.load(mask_file)

for smooth in random_smooths:
        
    for iteration in range(100):
        
        print smooth, iteration

        _,_,t1_left = read_vtk(random_file%(smooth, 'lh', str(iteration)))
        _,_,t1_right = read_vtk(random_file%(smooth, 'rh', str(iteration)))
        t1 = np.concatenate((t1_left, t1_right))
        
        # unmask embed which has been saved in masked form
        idcs=np.arange(0,t1.shape[0])
        nonmask=np.delete(idcs, mask)
        embed = np.zeros((t1.shape[0],100))
        embed[nonmask] = embed_masked
        
        # extend mask to nodes that have a value < 1500
        #bigmask = np.unique(np.concatenate((mask,np.where(t1<=1500)[0])))
        #bigmask = np.asarray(bigmask, dtype='int64')
        #nonmask_bigmask=np.delete(idcs, bigmask)
        bigmask = mask
        nonmask_bigmask = nonmask
        
        masked_t1 = np.delete(t1, bigmask)
        masked_embed = np.delete(embed, bigmask, axis=0)

        for m in range(len(all_maps)):
            
            maps = all_maps[m]
            maps_str = all_maps_str[m]
            print maps
        
            clf = linear_model.LinearRegression()
            clf.fit(masked_embed[:,maps], masked_t1)
            
            modelled_fit = clf.predict(masked_embed[:,maps])
            residuals = masked_t1 - clf.predict(masked_embed[:,maps])
            
            # write data back into cortex dimensions to avoid confusion
            modelled_fit_cortex = np.zeros((t1.shape[0])) 
            modelled_fit_cortex[nonmask_bigmask]=modelled_fit
            residuals_cortex = np.zeros((t1.shape[0]))  
            residuals_cortex[nonmask_bigmask]=residuals
            t1_cortex = np.zeros((t1.shape[0]))
            t1_cortex[nonmask_bigmask]=masked_t1
        
            
            model_dict = dict()
            model_dict['maps']=maps
            model_dict['coef']=clf.coef_
            model_dict['modelled_fit']=modelled_fit_cortex
            model_dict['residuals']=residuals_cortex
            model_dict['t1']=t1_cortex
            model_dict['score'] = clf.score(masked_embed[:,maps], masked_t1)
            model_dict['corr'] = stats.pearsonr(modelled_fit, masked_t1)
    
            print 'coeff', clf.coef_
            print 'score',model_dict['score']
            print 'corr', model_dict['corr']
            
            pkl_out = open(model_file%(iteration, smooth,maps_str), 'wb')
            pickle.dump(model_dict, pkl_out)
            pkl_out.close()
            
        
        
