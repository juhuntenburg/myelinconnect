from __future__ import division
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.spatial as spatial
from graphs import graph_from_mesh
from vtk_rw import read_vtk, write_vtk

subjects = pd.read_csv('/scr/ilz3/myelinconnect/subjects.csv')
subjects=list(subjects['DB'])
subjects.remove('KSMT')
hemis = ['lh', 'rh']
modes = ['',' ideal']

csv_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/qa/fixed/qa_fixed.csv'
edge_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/qa/fixed/%s_%s_edge_label.vtk'
no_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/qa/fixed/%s_%s_no_label.vtk'

columns = [sub+' '+hemi+mode for sub in subjects for hemi in hemis for mode in modes]
df = pd.DataFrame(columns=columns, index=['seeds min',
                                     'seeds max',
                                     'seeds mean',
                                     'seeds sdv',
                                     'labels min',
                                     'labels max',
                                     'labels mean',
                                     'labels sdv',
                                     'vertex no label',
                                     'label no vertex'])

for sub in subjects:
    for hemi in hemis:

        if os.path.isfile('/scr/ilz3/myelinconnect/all_data_on_simple_surf/labels_fixed/%s_%s_highres2lowres_labels.npy'%(sub, hemi)):
            
            print 'processing '+sub+' '+hemi
            
            # load data
            complex_v,complex_f, complex_d = read_vtk('/scr/ilz3/myelinconnect/struct/surf_%s/orig/mid_surface/%s_%s_mid.vtk'%(hemi, sub, hemi))
            simple_v, simple_f, simple_d = read_vtk('/scr/ilz3/myelinconnect/groupavg/indv_space/%s/lowres_%s_d_def.vtk'%(sub, hemi))
            labelling=np.load('/scr/ilz3/myelinconnect/all_data_on_simple_surf/labels_fixed/%s_%s_highres2lowres_labels.npy'%(sub, hemi))
            seeds=np.load('/scr/ilz3/myelinconnect/all_data_on_simple_surf/seeds_fixed/%s_%s_highres2lowres_seeds.npy'%(sub, hemi))

            # make an edge label surface and save it
            G = graph_from_mesh(complex_v, complex_f)
            edge_labelling = np.zeros_like(labelling[:,1])
            for v in range(edge_labelling.shape[0]):
                if any(labelling[G.neighbors(v),1] != labelling[v,1]):
                    edge_labelling[v] = 1
            write_vtk(edge_file%(sub, hemi), complex_v, complex_f, 
                      data=edge_labelling[:,np.newaxis])
            
            # make a no label surface and save it
            if np.where(labelling[:,1]==-1)[0].shape[0] >0:
                no_labelling = np.zeros_like(labelling[:,1])
                for n in list(np.where(labelling[:,1]==-1)[0]):
                    no_labelling[n] = 1
                write_vtk(no_file%(sub, hemi), complex_v, complex_f, 
                      data=no_labelling[:,np.newaxis])
            
            
            # get actual and ideal seed distances
            actual_seed_dist = np.linalg.norm((complex_v[seeds] - simple_v), axis=1)
            ideal_seed_dist, mapping  = spatial.KDTree(complex_v).query(simple_v)
            
            # count how many of each label
            label_count=np.empty((int(labelling[:,1].max()+1),))
            for i in range(int(labelling[:,1].max())+1):
               label_count[i] = len(np.where(labelling[:,1]==i)[0])
            
            # write values into data file
            df[sub+' '+hemi]['seeds min'] = np.min(actual_seed_dist)
            df[sub+' '+hemi]['seeds max'] = np.max(actual_seed_dist)
            df[sub+' '+hemi]['seeds mean'] = np.mean(actual_seed_dist)
            df[sub+' '+hemi]['seeds sdv'] = np.std(actual_seed_dist)
            df[sub+' '+hemi+' ideal']['seeds min'] = np.min(ideal_seed_dist)
            df[sub+' '+hemi+' ideal']['seeds max'] = np.max(ideal_seed_dist)
            df[sub+' '+hemi+' ideal']['seeds mean'] = np.mean(ideal_seed_dist)
            df[sub+' '+hemi+' ideal']['seeds sdv'] = np.std(ideal_seed_dist)
            df[sub+' '+hemi]['labels min']= np.min(label_count)
            df[sub+' '+hemi]['labels max']= np.max(label_count)
            df[sub+' '+hemi]['labels mean']= np.mean(label_count)
            df[sub+' '+hemi]['labels sdv']= np.std(label_count)
            df[sub+' '+hemi]['vertex no label']= np.where(labelling[:,1]==-1)[0].shape[0]
            df[sub+' '+hemi]['label no vertex']= np.where(label_count==0.0)[0].shape[0]
            df[sub+' '+hemi+' ideal']['labels min']= (complex_v.shape[0] / simple_v.shape[0])
            df[sub+' '+hemi+' ideal']['labels max']= (complex_v.shape[0] / simple_v.shape[0])
            df[sub+' '+hemi+' ideal']['labels mean']= (complex_v.shape[0] / simple_v.shape[0])
            df[sub+' '+hemi+' ideal']['labels sdv']= 0
            df[sub+' '+hemi+' ideal']['vertex no label'] = 0
            df[sub+' '+hemi+' ideal']['label no vertex']= 0
        
        else:
            print 'not existing '+sub+' '+hemi
        
# save csv
df.to_csv(csv_file)