import numpy as np
import pickle


corr_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/corr/rh_smooth_3_avg_corr.hdf5'
windows_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/path/windows_longpath3_rad4.pkl'
# see notebook path_embedding_initial
corr_dict_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/path/newcorr_windows_longpath3_rad4.pkl'


f = h5py.File(corr_file, 'r')
full_shape=tuple(f['shape'])
upper_corr=np.asarray(f['upper'])
f.close()
full_corr = np.zeros(tuple(full_shape))
full_corr[np.triu_indices_from(full_corr, k=1)] = np.nan_to_num(upper_corr)
full_corr += full_corr.T



with open(windows_file, 'r') as pkl_in:
    windows = pickle.load(pkl_in)
flat_windows = []
for w in windows:
    flat_windows+=[x for x in w]
nodes=list(np.unique(flat_windows))

corr_dict = {}
for node in nodes:
    corr_dict[node]=full_corr[node]
with open(corr_dict_file, 'w') as pkl_out:
    pickle.dump(corr_dict, pkl_out)