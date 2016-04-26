import numpy as np
from vtk_rw import read_vtk
from sklearn import linear_model
import scipy.stats as stats
import pickle



mesh_file = '/scr/ilz3/myelinconnect/new_groupavg/surfs/lowres/%s_lowres_new.vtk'
mask_file = '/scr/ilz3/myelinconnect/new_groupavg/masks/%s_fullmask.npy'
random_file = '/scr/ilz3/myelinconnect/new_groupavg/model/random_data/%s/%s_random_normal_%s_smoothdata.vtk'
embed_dict_file = '/scr/ilz3/myelinconnect/new_groupavg/embed/%s_%s_embed_dict.pkl'
model_file = '/scr/ilz3/myelinconnect/new_groupavg/model/linear_combination/random/random_%s_%s_%s_t1avg_by_fc_maps_%s.pkl'


maps = range(10)
maps_str = '0_to_10'

embed_smooth='smooth_3'
random_smooths=['smooth_3', 'smooth_6', 'smooth_12', 'smooth_20']
hemis = ['lh', 'rh']

for smooth in random_smooths: 
    for hemi in hemis:
        
        for iteration in range(100):
            
            print smooth, hemi#, iteration
    
            v,f,d = read_vtk(mesh_file%hemi)
            _,_,t1 = read_vtk(random_file%(smooth, hemi, iteration))
            mask = np.load(mask_file%hemi)
            
            # prepare embedding (normalized from entry in dict)
            pkl_in = open(embed_dict_file%(hemi,embed_smooth), 'r')
            embed_dict=pickle.load(pkl_in)
            pkl_in.close()
            
            embed_masked = np.zeros((embed_dict['vectors'].shape[0], embed_dict['vectors'].shape[1]-1))
            for comp in range(10):
                embed_masked[:,comp]=(embed_dict['vectors'][:,comp+1]/embed_dict['vectors'][:,0])
            
            idcs=np.arange(0,v.shape[0])
            nonmask=np.delete(idcs, mask)
            embed = np.zeros((v.shape[0],10))
            embed[nonmask] = embed_masked
            
            
            # extend mask to nodes that have a t1avg < 1500
            #mask = np.unique(np.concatenate((mask,np.where(t1<=1500)[0])))
            #mask = np.asarray(mask, dtype='int64')
            #nonmask_bigmask=np.delete(idcs, mask)
            
            masked_t1 = np.delete(t1, mask)
            masked_embed = np.delete(embed, mask, axis=0)
            
            #t1_norm = (masked_t1 - np.mean(masked_t1)) / (np.std(masked_t1))
            #embed_norm = np.zeros_like(masked_embed)
            #for comp in range(masked_embed.shape[1]):
            #    embed_norm[:,comp] = (scatter_embed[:,comp] - np.mean(masked_embed[:,comp])) / (np.std(masked_embed[:,comp]))
            
            clf = linear_model.LinearRegression()
            clf.fit(masked_embed[:,maps], masked_t1)
            modelled_fit = np.dot(masked_embed[:,maps], clf.coef_)
            modelled_fit_norm = (modelled_fit - np.mean(modelled_fit)) / np.std(modelled_fit)
            
            t1_norm = (masked_t1 - np.mean(masked_t1)) / (np.std(masked_t1))
            residuals = t1_norm - modelled_fit_norm
            
            
            # write data back into cortex dimensions to avoid confusion
            modelled_fit_cortex = np.zeros((v.shape[0])) 
            modelled_fit_cortex[nonmask]=modelled_fit
            modelled_fit_norm_cortex = np.zeros((v.shape[0])) 
            modelled_fit_norm_cortex[nonmask]=modelled_fit_norm 
            residuals_cortex = np.zeros((v.shape[0]))  
            residuals_cortex[nonmask]=residuals
            t1_norm_cortex = np.zeros((v.shape[0])) 
            t1_norm_cortex[nonmask]=t1_norm
            
            
            model_dict = dict()
            model_dict['maps']=maps
            model_dict['coef']=clf.coef_
            model_dict['t1_norm']=t1_norm_cortex
            model_dict['modelled_fit']=modelled_fit_cortex
            model_dict['modelled_fit_norm']=modelled_fit_norm_cortex
            model_dict['residuals']=residuals_cortex
            model_dict['score'] = clf.score(masked_embed[:,maps], masked_t1)
            model_dict['corr'] = stats.pearsonr(modelled_fit, masked_t1)
            
            print hemi
            print maps
            print 'coeff', clf.coef_
            print 'score',model_dict['score']
            print 'corr', model_dict['corr']
            
            pkl_out = open(model_file%(iteration, hemi, smooth, maps_str), 'wb')
            pickle.dump(model_dict, pkl_out)
            pkl_out.close()
            
        
        
