from __future__ import division
import numpy as np
from vtk_rw import read_vtk, write_vtk
import pandas as pd
import sys


#subjects = pd.read_csv('/scr/ilz3/myelinconnect/subjects.csv')
#subjects=list(subjects['DB'])
#subjects.remove('KSMT')

subjects = sys.argv[1]
subjects = [subjects]

hemis = ['rh']
sessions = ['1_1', '1_2', '2_1', '2_2']

rest_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/rest/%s_%s_rest%s.npy'
corr_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/corr/%s_%s_rest%s_corr.npy'
avg_corr_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/corr/%s_%s_avg_corr.npy'
sub_avg_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/corr/%s_allsub_avg_corr.npy'

for hemi in hemis:
    
    # figure out size of the resulting matrix (N_verticesxN_vertices)
    get_size = np.load('/scr/ilz3/myelinconnect/all_data_on_simple_surf/rest/%s_%s_rest%s.npy'%(subjects[0], hemi, sessions[0])).shape[0]
    
    # create empty matrix for cross subject average
    sub_avg = np.zeros((get_size, get_size))
    
    sub_count = 0
    for sub in subjects:
        
        # find out if correlation matrix has been produced before and load
        try:
            avg_corr = np.load(avg_corr_file%(sub, hemi))
            
        # if not, create avg correlation matrix over all 4 runs
        except IOError:
            
            # make empty matrix
            avg_corr_z = np.zeros((get_size, get_size))
            
            sess_count = 0
            for sess in sessions:
            
                # find out if intermediate corr matrix has been saved
                # (more or less deprecated as it takes too much disk space)
                try:
                    corr = np.load(corr_file%(sub, hemi, sess))
                
                # otherwise load resting state time series of one run and 
                # create correlation matrix
                except IOError:
                    
                    rest = np.load(rest_file%(sub, hemi, sess))
                    corr = np.nan_to_num(np.corrcoef(rest))
                    del rest
                    #np.save(corr_file%(sub, hemi, sess), corr)
                
                # fisher r-to-z transform correlations
                corr_z = np.arctanh(corr)
                del corr
                # add to cross-run average matrix
                avg_corr_z += corr_z
                del corr_z
                
                # increase count for later division
                sess_count += 1
                print 'Finished '+sess
            
            # after all sessions have been processed, divide average by count
            avg_corr_z = avg_corr_z / sess_count
            # tranfroms back to r values
            avg_corr = np.tanh(avg_corr_z)
            del avg_corr_z
            # and save
            np.save(avg_corr_file%(sub, hemi), avg_corr)
            #del avg_corr
            
            # threshold to maintain only top 10% of correlations
            avg_corr_thr =
            
            # add to cross subject average
            sub_avg += avg_corr_thr
            
            # increase sub count for division
            sub_count += 1
            
        sub_avg = sub_avg / sub_count
        np.save(sub_avg_file%(hemi), sub_avg)
        
        
        