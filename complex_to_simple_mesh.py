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


# function returning generator for sub,hemi tuples fro joblib
def tupler(subjects, hemis):
    for s in subjects:
        for h in hemis:
            yield (s, h)

# function to find unlabelled neighbours of a node in the graph
# and add them to the tree correctly
def add_neighbours(node, length, graph, labels, tree):
    # find direct neighbours of the node
    neighbours = np.array(graph.neighbors(node))
    # check that they don't already have a label
    unlabelled = neighbours[np.where(labels[neighbours][:,1]==-1)[0]]
    # insert source neighbour pair with edge length to tree
    for u in unlabelled:
        new_length = length + graph[node][u]['length']
        tree.insert(new_length,(node, u))
    
    return tree

# function to write time and message to log file
def log(log_file, message, logtime=True):
    with open(log_file, 'a') as f:
        if logtime:
            f.write(time.ctime()+'\n')
        f.write(message+'\n')


# main function (might be good to split eventually)
def create_mapping((sub, hemi)):

    
    # log different steps in logfile
    log_file = '/scr/ilz3/myelinconnect/working_dir/complex_to_simple/AVL_log_worker_%s.txt'%(str(os.getpid()))
    log(log_file, 'Processing %s %s'%(sub, hemi))

    # not optimal to have this hardcoded inside the function
    # figure out how to pipe with parallel and move outside
    complex_file = '/scr/ilz3/myelinconnect/struct/surf_%s/orig/mid_surface/%s_%s_mid.vtk'
    simple_file = '/scr/ilz3/myelinconnect/groupavg/indv_space/%s/lowres_%s_d_def.vtk'# version d
    label_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/labels/AVL/%s_%s_highres2lowres_labels.npy'
    surf_label_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/labels/AVL/%s_%s_highres2lowres_labels.vtk'

    # load the meshes
    log(log_file, '...loading data', logtime=False)
    complex_v, complex_f, complex_d = read_vtk(complex_file%(hemi, sub, hemi))
    simple_v, simple_f, simple_d = read_vtk(simple_file%(sub, hemi))

    # find those points on the individuals complex mesh that correspond best
    # to the simplified group mesh in subject space
    log(log_file, '...running kdtree')
    inaccuracy, mapping = spatial.KDTree(complex_v).query(simple_v)
    # find coordinates of those points in the highres mesh
    voronoi_seed_idx = mapping
    voronoi_seed_coord = complex_v[mapping]

    # double check differences
    log(log_file, '...checking kdtree')
    dist = np.linalg.norm((voronoi_seed_coord - simple_v), axis=1)
    if any(dist != inaccuracy):
        sys.exit("simple to complex mesh mapping not correct")

    # convert highres mesh into graph containing edge length
    log(log_file, '...creating graph')
    complex_graph = graph_from_mesh(complex_v, complex_f, edge_length=True)

    # make a labelling container to be filled with the search tree
    # first column are the vertex indices of the complex mesh
    # second column are the labels from the simple mesh
    # (-1 for all but the corresponding points for now)
    log(log_file, '...initiating labels')
    labelling = np.zeros((complex_v.shape[0],2), dtype='int64')-1
    labelling[:,0] = range(complex_v.shape[0])

    for i in range(voronoi_seed_idx.shape[0]):
        labelling[voronoi_seed_idx[i]][1] = i

    # initiate AVLTree for binary search
    log(log_file, '...building search tree')
    tree = FastAVLTree()
    # organisation of the tree will be
    # key: edge length
    # value: tuple of vertices (source, target)

    # find all neighbours of the voronoi seeds
    for v in voronoi_seed_idx:
        add_neighbours(v, 0, complex_graph, labelling, tree)

    # Competetive fast marching starting from voronoi seeds
    log(log_file, '...marching')
    while any(labelling[:,1]==-1):
        while tree.count > 0:
            # pop the item with minimum edge length
            min_item = tree.pop_min()
            length = min_item[0]
            source = min_item[1][0]
            target = min_item[1][1]

            #if target no label yet (but source does!), assign label of source
            if labelling[target][1] == -1:
                if labelling[source][1] == -1:
                    sys.exit('Source has no label, something went wrong!')
                else:
                    # assign label of source to target
                    labelling[target][1] = labelling[source][1]

            # add neighbours of target to tree
            add_neighbours(target, length, complex_graph, labelling, tree)

    # write out labelling file and surface with labels
    log(log_file, '...saving data')
    np.save(label_file%(sub, hemi), labelling)
    write_vtk(surf_label_file%(sub, hemi), complex_v, complex_f,
                data=labelling[:,1, np.newaxis])

    log(log_file, 'Finished %s %s'%(sub, hemi))

    return log_file


if __name__ == "__main__":

    #cachedir = '/scr/ilz3/myelinconnect/working_dir/complex_to_simple/'
    #memory = Memory(cachedir=cachedir)
    
    subjects = pd.read_csv('/scr/ilz3/myelinconnect/subjects.csv')
    subjects=list(subjects['DB'])
    subjects.remove('KSMT')
    hemis = ['lh', 'rh']

    Parallel(n_jobs=16)(delayed(create_mapping)(i) 
                               for i in tupler(subjects, hemis))
