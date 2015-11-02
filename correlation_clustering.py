from __future__ import division
import numpy as np
#import numexpr as ne
#ne.set_num_threads(ne.ncores)
import pandas as pd
#import hcp_corr
import h5py


subjects = pd.read_csv('/scr/ilz3/myelinconnect/subjects.csv')
subjects=list(subjects['DB'])
subjects.remove('KSMT')

smooths=['2'] #,'raw', '3']
hemis = ['rh']
sessions = ['1_1', '1_2' , '2_1', '2_2']


rest_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/rest/smooth_%s/%s_%s_rest%s_smooth_%s.npy'
thr_corr_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/corr/%s_smooth_%s_thr_per_session_corr.hdf5'
corr_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/corr/%s_smooth_%s_avg_corr.hdf5'
#rest_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/rest/%s/%s_%s_rest%s.npy'
#thr_corr_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/corr/%s_%s_thr_per_session_corr.hdf5'
#corr_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/corr/%s_%s_avg_corr.hdf5'


for smooth in smooths: 
    
    print 'smooth '+smooth

    for hemi in hemis:
        
        print 'hemi '+hemi
    
        # figure out size of the resulting matrix (N_verticesxN_vertices)
        get_size = np.load(rest_file%(smooth, subjects[0], hemis[0], sessions[0], smooth)).shape[0]
        #get_size = np.load(rest_file%(smooth, subjects[0], hemis[0], sessions[0])).shape[0]
        
        # create empty vector for cross subject average / thresholded
        if np.mod((get_size**2-get_size),2)==0.0:
            avg_corr = np.zeros((get_size**2-get_size)/2)
            avg_corr_thr = np.zeros((get_size**2-get_size)/2)
        
        else:
            print 'size calculation no zero mod'
        
        # session count
        count = 0
        
        for sub in subjects:
            
            for sess in sessions:
                
                print sub, sess
                
                # load time series
                rest = np.load(rest_file%(smooth, sub, hemi, sess, smooth))
                #rest = np.load(rest_file%(smooth, sub, hemi, sess))
                
                # calculate correlations matrix
                #K = hcp_corr.corrcoef_upper(rest)
                corr = np.nan_to_num(np.corrcoef(rest))
                del rest
                
                # get upper triangular only
                corr = corr[np.triu_indices_from(corr, k=1)]
                
                # r-to-z trans and add to avg
                avg_corr += np.arctanh(corr)
                
    
                # threshold and add to thresholded avg
                thr = np.percentile(corr, 90)
                avg_corr_thr[np.where(corr>thr)] += 1

                del corr
                
                # increase count
                count += 1
                
        
        # transform back to r and divide by number of sessions included
        avg_corr /= count
        avg_corr = np.nan_to_num(np.tanh(avg_corr))
        
        avg_corr_thr /= count
        
        # save upper triangular correlation matrix as hdf5
        f = h5py.File(corr_file%(hemi, smooth), 'w')
        f.create_dataset('upper', data=avg_corr)
        f.create_dataset('shape', data=(get_size, get_size))
        f.close()
        
        f_thr = h5py.File(thr_corr_file%(hemi, smooth), 'w')
        f_thr.create_dataset('upper', data=avg_corr_thr)
        f_thr.create_dataset('shape', data=(get_size, get_size))
        f_thr.close()      
        
        
        del avg_corr
        del avg_corr_thr  