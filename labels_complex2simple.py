from __future__ import division
import numpy as np
import scipy.spatial as spatial
import sys
import time
import os
import pandas as pd
from joblib import Memory, Parallel, delayed
# from https://pypi.python.org/pypi/bintrees/2.0.2
from bintrees import FastAVLTree, FastBinaryTree
# from https://github.com/juhuntenburg/brainsurfacescripts
from vtk_rw import read_vtk, write_vtk
from graphs import graph_from_mesh
from simplification import add_neighbours, find_voronoi_seeds, competetive_fast_marching
from utils import log, tupler
import pdb


'''
Maps vertices from a simplified mesh to the closest corresponding
vertices of the original mesh using a KDTree. These vertices are
then used as seeds for a Voronoi tesselation of the complex mesh which is
implemented as a competitive fast marching in a balanced binary (AVL)tree.
A mapping is created which associates each vertex of the complex mesh
with the closest vertex of the simple mesh.
-----------------------------------------------------------------------------
Binary search trees: https://en.wikipedia.org/wiki/Binary_search_tree
Balanced binary trees: https://en.wikipedia.org/wiki/AVL_tree,
Using them as heaps: http://samueldotj.com/blog/?s=binary+tree
Implementation used: https://pypi.python.org/pypi/bintrees/2.0.2 (cython version)
'''    

# main function for looping over subject and hemispheres
def create_mapping((sub, hemi)):
    
    print sub, hemi

    complex_file = '/scr/ilz3/myelinconnect/struct/surf_%s/orig/mid_surface/%s_%s_mid.vtk'
    simple_file = '/scr/ilz3/myelinconnect/groupavg/indv_space/%s/lowres_%s_d_def.vtk'# version d
    
    log_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/labels_ideal/logs/log_worker_%s.txt'%(str(os.getpid()))
    seed_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/seeds_ideal/%s_%s_highres2lowres_seeds.npy'
    label_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/labels_ideal/%s_%s_highres2lowres_labels.npy'
    surf_label_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/labels_ideal/%s_%s_highres2lowres_labels.vtk'

    # load the meshes
    log(log_file, 'Processing %s %s'%(sub, hemi))
    log(log_file, '...loading data', logtime=False)
    complex_v, complex_f, complex_d = read_vtk(complex_file%(hemi, sub, hemi))
    simple_v, simple_f, simple_d = read_vtk(simple_file%(sub, hemi))

    # find those points on the individuals complex mesh that correspond best
    # to the simplified group mesh in subject space
    log(log_file, '...finding unique voronoi seeds')
    
    try:
        voronoi_seed_idx = np.load(seed_file%(sub, hemi))
        voronoi_seed_coord = complex_v[voronoi_seed_idx]
    
    except IOError:
    
        voronoi_seed_idx, inaccuracy, log_file  = find_voronoi_seeds(simple_v, 
                                                                     simple_f, 
                                                                     complex_v, 
                                                                     complex_f,
                                                           log_file = log_file)
        np.save(seed_file%(sub, hemi), voronoi_seed_idx)
        
        # find coordinates of those points in the highres mesh
        voronoi_seed_coord = complex_v[voronoi_seed_idx]
        
        # double check differences
        log(log_file, '...checking unique vs nearest mapping')
        dist = np.linalg.norm((voronoi_seed_coord - simple_v), axis=1)
        if ((np.mean(dist)-np.mean(inaccuracy[:,0])>0.1)):
            log(log_file, 'Unique seeds very far from nearest seeds! %f'%(np.mean(dist)-np.mean(inaccuracy[:,0])))

    # convert highres mesh into graph containing edge length
    log(log_file, '...creating graph')
    complex_graph = graph_from_mesh(complex_v, complex_f, edge_length=True)

    # find the actual labels
    log(log_file, '...competetive fast marching')
    labels = competetive_fast_marching(complex_v, complex_graph, voronoi_seed_idx)

    # write out labelling file and surface with labels
    log(log_file, '...saving data')
    np.save(label_file%(sub, hemi), labels)
    write_vtk(surf_label_file%(sub, hemi), complex_v, complex_f,
                data=labels[:,1, np.newaxis])

    log(log_file, 'Finished %s %s'%(sub, hemi))

    return log_file



#create_mapping(('BP4T','rh'))

if __name__ == "__main__":

    #cachedir = '/scr/ilz3/myelinconnect/working_dir/complex_to_simple/'
    #memory = Memory(cachedir=cachedir)
    
    subjects = pd.read_csv('/scr/ilz3/myelinconnect/subjects.csv')
    subjects=list(subjects['DB'])
    subjects.remove('KSMT')
    hemis = ['rh', 'lh']
    

    Parallel(n_jobs=16)(delayed(create_mapping)(i) 
                               for i in tupler(subjects, hemis))
