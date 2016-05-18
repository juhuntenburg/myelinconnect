import numpy as np
from vtk_rw import read_vtk
from sklearn import linear_model
import scipy.stats as stats
import pickle
from histogram_matching import hist_match



#mesh_file = '/scr/ilz3/myelinconnect/new_groupavg/surfs/lowres/%s_lowres_new.vtk'
t1_file = '/scr/ilz3/myelinconnect/new_groupavg/t1/smooth_3/%s_t1_avg_smooth_3.npy'
mask_file="/scr/ilz3/myelinconnect/new_groupavg/masks/%s_fullmask_new.npy"
fullmask_file = '/scr/ilz3/myelinconnect/new_groupavg/masks/fullmask_lh_rh_new.npy'
random_file = '/scr/ilz3/myelinconnect/new_groupavg/model/random_data/%s/%s_random_normal_%s_smoothdata.vtk'
embed_dict_file="/scr/ilz3/myelinconnect/new_groupavg/embed/both_smooth_3_embed_dict.pkl"
model_file = '/scr/ilz3/myelinconnect/new_groupavg/model/linear_combination/random/%s_random_%s_%s_t1avg_by_fc_maps_%s.pkl'


all_maps = [[0],range(10)]
all_maps_str = ['0', '0_to_10']
random_smooths=['smooth_3', 'smooth_6', 'smooth_12', 'smooth_20']

# load t1 file for dimensions and histogramm
t1_left = np.load(t1_file%('lh'))
t1_right = np.load(t1_file%('rh'))

# prepare embedding (normalized from entry in dict)
pkl_in = open(embed_dict_file, 'r')
embed_dict=pickle.load(pkl_in)
pkl_in.close()

embed_masked = np.zeros((embed_dict['vectors'].shape[0], embed_dict['vectors'].shape[1]-1))
for comp in range(100):
    embed_masked[:,comp]=(embed_dict['vectors'][:,comp+1]/embed_dict['vectors'][:,0])

fullmask = np.load(fullmask_file)
# unmask embed which has been saved in masked form
idcs_full=np.arange(0,(t1_left.shape[0]+t1_right.shape[0]))
nonmask_full=np.delete(idcs_full, fullmask)
embed_full = np.zeros(((t1_left.shape[0]+t1_right.shape[0]),100))
embed_full[nonmask_full] = embed_masked

for hemi in ['lh', 'rh']:
    
    print hemi
    if hemi == 'lh':
        embed = embed_full[:t1_left.shape[0]]
        idcs_small = np.arange(0,t1_left.shape[0])
        t1_orig = np.copy(t1_left)
    elif hemi =='rh':
        embed = embed_full[t1_left.shape[0]:]
        idcs_small = np.arange(0,t1_right.shape[0])
        t1_orig = np.copy(t1_right)
        
    mask = np.load(mask_file%hemi)
    #extend mask to regions < 1500
    mask = np.unique(np.concatenate((mask, np.where(t1_orig<=1500)[0])))
    nonmask_small = np.delete(idcs_small, mask)
    
    # mask and normalize embedding vectors to norm=1
    masked_embed = np.delete(embed, mask, axis=0)
    masked_embed /= np.linalg.norm(masked_embed, axis=0)

    for smooth in random_smooths:
            
        for iteration in range(1000):
            
            print smooth, iteration, hemi

            _,_,t1 = read_vtk(random_file%(smooth, hemi, str(iteration)))
            masked_t1 = np.delete(t1, mask)
            masked_orig = np.delete(t1_orig, mask)
            
            # histogramm matching to original t1
            masked_t1 = hist_match(masked_t1, masked_orig)
    
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
                
                pkl_out = open(model_file%(hemi, iteration, smooth,maps_str), 'wb')
                pickle.dump(model_dict, pkl_out)
                pkl_out.close()
                
            
            
