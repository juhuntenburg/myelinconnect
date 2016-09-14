from __future__ import division
import numpy as np
import numexpr as ne
import pandas as pd
import h5py
import pickle
import hcp_corr 
from mapalign import embed

'''
Starting from individual subject time series data in group average space,
create first individual and then group average connectivity matrix, 
mask and embed.
'''


'''
--------------------
INPUTS AND SETTINGS
--------------------
'''
ne.set_num_threads(ne.ncores-1)

all_subjects = pd.read_csv('/scr/ilz3/myelinconnect/subjects.csv')
all_subjects=list(all_subjects['DB'])
all_subjects.remove('KSMT')

smooths=['smooth_3'] 
sessions = ['1_1', '1_2' , '2_1', '2_2']
n_embedding = 100

rest_file = '/scr/ilz3/myelinconnect/new_groupavg/rest/smooth_3/%s_%s_rest%s_smooth_3.npy'
mask_file="/scr/ilz3/myelinconnect/new_groupavg/masks/fullmask_lh_rh_new.npy"
corr_file = '/scr/ilz3/myelinconnect/new_groupavg/corr/both_smooth_3_avg_corr.hdf5'
embed_file="/scr/ilz3/myelinconnect/new_groupavg/embed/both_smooth_3_embed.npy"
embed_dict_file="/scr/ilz3/myelinconnect/new_groupavg/embed/both_smooth_3_embed_dict.pkl"

calc_corr = False
save_corr = False
calc_embed = True


'''
----------
FUNCTIONS
---------
'''
def avg_correlation(ts_files, thr=None):
    '''
    Calculates average connectivity matrix using hcp_corr package for memory 
    optimization: https://github.com/NeuroanatomyAndConnectivity/hcp_corr
    '''
    # make empty avg corr matrix
    if type(ts_files[0])==str:
        get_size = np.load(ts_files[0]).shape[0]
    elif type(ts_files[0])==np.ndarray:
        get_size = ts_files[0].shape[0]
        
    full_shape = (get_size, get_size)
    if np.mod((get_size**2-get_size),2)==0.0:
        avg_corr = np.zeros((get_size**2-get_size)/2)
    else:
        print 'size calculation no zero mod'

    count = 0
    for rest in ts_files:
        # load time series
        if type(rest)==str:
            rest = np.load(rest)
        elif type(rest)==np.ndarray:
            pass
        # calculate correlations matrix
        print '...corrcoef'
        corr = hcp_corr.corrcoef_upper(rest)
        del rest
        # threshold / transform
        if thr == None:
            # r-to-z transform and add to avg
            print '...transform'
            avg_corr += ne.evaluate('arctanh(corr)')
        else:
            # threshold and add to avg
            print '...threshold'
            thr = np.percentile(corr, 100-thr)
            avg_corr[np.where(corr>thr)] += 1
        del corr
        count += 1
    # divide by number of sessions included
    print '...divide'
    avg_corr /= count
    # transform back if necessary
    if thr == None:
        print '...back transform'
        avg_corr = np.nan_to_num(ne.evaluate('tanh(avg_corr)'))

    return avg_corr, full_shape

def recort(n_vertices, data, cortex, increase):
    '''
    Helper function to rewrite masked, embedded data to full cortex
    (copied from Daniel Margulies)
    '''
    d = np.zeros(n_vertices)
    count = 0
    for i in cortex:
        d[i] = data[count] + increase
        count = count +1
    return d

def embedding(upper_corr, full_shape, mask, n_components):
    '''
    Diffusion embedding on connectivity matrix using mapaling package:
    https://github.com/satra/mapalign
    '''
    # reconstruct full matrix
    print '...full matrix'
    full_corr = np.zeros(tuple(full_shape))
    full_corr[np.triu_indices_from(full_corr, k=1)] = np.nan_to_num(upper_corr)
    full_corr += full_corr.T
    all_vertex=range(full_corr.shape[0])

    # apply mask
    print '...mask'
    masked_corr = np.delete(full_corr, mask, 0)
    del full_corr
    masked_corr = np.delete(masked_corr, mask, 1)
    cortex=np.delete(all_vertex, mask)

    # run actual embedding
    print '...embed'
    K = (masked_corr + 1) / 2.
    del masked_corr
    K[np.where(np.eye(K.shape[0])==1)]=1.0
    #v = np.sqrt(np.sum(K, axis=1))
    #A = K/(v[:, None] * v[None, :])
    #del K, v
    #A = np.squeeze(A * [A > 0])
    #embedding_results = runEmbed(A, n_components_embedding)
    embedding_results, embedding_dict = embed.compute_diffusion_map(K, n_components=n_components, overwrite=True)

    # reconstruct masked vertices as zeros
    embedding_recort=np.zeros((len(all_vertex),embedding_results.shape[1]))
    for e in range(embedding_results.shape[1]):
        embedding_recort[:,e]=recort(len(all_vertex), embedding_results[:,e], cortex, 0)

    return embedding_recort, embedding_dict


'''
----
RUN
----
'''
    
'''avg correlations'''
if calc_corr:
    ts_files = []
    for sub in subjects:
        for sess in sessions:
            rest_left = np.load(rest_file%(sub, 'lh', sess))
            rest_right = np.load(rest_file%(sub, 'rh', sess))
            ts_file = np.concatenate((rest_left, rest_right))
            ts_files.append(ts_file)

    print 'calculating average correlations'
    upper_corr, full_shape = avg_correlation(ts_files)

    if save_corr:
        print 'saving matrix'
        f = h5py.File(corr_file, 'w')
        f.create_dataset('upper', data=upper_corr)
        f.create_dataset('shape', data=full_shape)
        f.close()

    if (not calc_embed):
        del upper_corr

'''embedding'''
if calc_embed:
    print 'embedding'

    if not calc_corr:
        print '...load'
        # load upper triangular avg correlation matrix
        f = h5py.File(corr_file, 'r')
        upper_corr = np.asarray(f['upper'])
        full_shape = tuple(f['shape'])
        f.close()

    mask = np.load(mask_file)
    embedding_recort, embedding_dict = embedding(upper_corr, full_shape, mask, n_embedding)

    np.save(embed_file,embedding_recort)
    pkl_out = open(embed_dict_file, 'wb')
    pickle.dump(embedding_dict, pkl_out)
    pkl_out.close()
