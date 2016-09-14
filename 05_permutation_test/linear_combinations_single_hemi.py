import numpy as np
from vtk_rw import read_vtk
from sklearn import linear_model
import scipy.stats as stats
import pickle

'''
Rerun the fitting of T1 with linear combinations of embedding components 
(FC1 and FC1,5,6) for each hemisphere separately for comparison to random data.
'''

mesh_file = '/scr/ilz3/myelinconnect/new_groupavg/surfs/lowres/%s_lowres_new.vtk'
mask_file="/scr/ilz3/myelinconnect/new_groupavg/masks/%s_fullmask_new.npy"
fullmask_file = '/scr/ilz3/myelinconnect/new_groupavg/masks/fullmask_lh_rh_new.npy'
t1_file = '/scr/ilz3/myelinconnect/new_groupavg/t1/smooth_1.5/%s_t1_avg_smooth_1.5.npy'
embed_dict_file="/scr/ilz3/myelinconnect/new_groupavg/embed/both_smooth_3_embed_dict.pkl"
model_file = '/scr/ilz3/myelinconnect/new_groupavg/model/linear_combination/t1avg/smooth_1.5/%s_t1avg_by_fc_maps_%s.pkl'


all_maps = [[0],[0,4,5]]
all_maps_str = ['0', 'best']

# load one file for each hemishpere once to get dimensions
lv,_,_ = read_vtk(mesh_file%('lh'))
rv,_,_ = read_vtk(mesh_file%('rh'))

# prepare embedding (normalized from entry in dict)
pkl_in = open(embed_dict_file, 'r')
embed_dict=pickle.load(pkl_in)
pkl_in.close()

embed_masked = np.zeros((embed_dict['vectors'].shape[0], embed_dict['vectors'].shape[1]-1))
for comp in range(100):
    embed_masked[:,comp]=(embed_dict['vectors'][:,comp+1]/embed_dict['vectors'][:,0])

fullmask = np.load(fullmask_file)
# unmask embed which has been saved in masked form
idcs_full=np.arange(0,(lv.shape[0]+rv.shape[0]))
nonmask_full=np.delete(idcs_full, fullmask)
embed_full = np.zeros(((lv.shape[0]+rv.shape[0]),100))
embed_full[nonmask_full] = embed_masked

for hemi in ['lh', 'rh']:
    
    print hemi
    if hemi == 'lh':
        embed = embed_full[:lv.shape[0]]
        idcs_small = np.arange(0,lv.shape[0])
    elif hemi =='rh':
        embed = embed_full[lv.shape[0]:]
        idcs_small = np.arange(0,rv.shape[0])
        
    mask = np.load(mask_file%hemi)
    nonmask_small = np.delete(idcs_small, mask)
    masked_embed = np.delete(embed, mask, axis=0)


    t1 = np.load(t1_file%(hemi))
    masked_t1 = np.delete(t1, mask)

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

        print 'coeff', clf.coef_
        print 'score',model_dict['score']
        print 'corr', model_dict['corr']
        
        pkl_out = open(model_file%(hemi, maps_str), 'wb')
        pickle.dump(model_dict, pkl_out)
        pkl_out.close()
                
            
            
