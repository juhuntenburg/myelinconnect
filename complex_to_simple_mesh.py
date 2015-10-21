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

# function to write time and message to log file
def log(log_file, message, logtime=True):
    with open(log_file, 'a') as f:
        if logtime:
            f.write(time.ctime()+'\n')
        f.write(message+'\n')

def find_voronoi_seeds(simple_vertices, complex_vertices):
    '''
    Finds those points on the complex mehs that correspoind best to the
    simple mesh while forcing a one-to-one mapping
    '''
    # make array for writing in final voronoi seed indices
    voronoi_seed_idx = np.zeros((simple_vertices.shape[0],), dtype='int64')-1
    missing = np.where(voronoi_seed_idx==-1)[0].shape[0]
    mapping_single = np.zeros_like(voronoi_seed_idx)

    neighbours = 0
    col = 0

    while missing != 0:

        neighbours += 100
        # find nearest neighbours
        inaccuracy, mapping  = spatial.KDTree(complex_vertices).query(simple_vertices, k=neighbours)
        # go through columns of nearest neighbours until unique mapping is
        # achieved, if not before end of neighbours, extend number of neighbours
        while col < neighbours:
            # find all missing voronoi seed indices
            missing_idx = np.where(voronoi_seed_idx==-1)[0]
            missing = missing_idx.shape[0]
            if missing == 0:
                break
            else:
                # for missing entries fill in next neighbour
                mapping_single[missing_idx]=np.copy(mapping[missing_idx,col])
                # find unique values in mapping_single
                unique, double_idx = np.unique(mapping_single, return_inverse = True)
                # empty voronoi seed index
                voronoi_seed_idx = np.zeros((simple_vertices.shape[0],), dtype='int64')-1
                # fill voronoi seed idx with unique values
                for u in range(unique.shape[0]):
                    # find the indices of this value in mapping
                    entries = np.where(double_idx==u)[0]
                    # set the first entry to the value
                    voronoi_seed_idx[entries[0]] = unique[u]
                # go to next column
                col += 1 
                
    return voronoi_seed_idx, inaccuracy


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


def competetive_fast_marching(vertices, graph, seeds):
    # make a labelling container to be filled with the search tree
    # first column are the vertex indices of the complex mesh
    # second column are the labels from the simple mesh
    # (-1 for all but the corresponding points for now)
    labels = np.zeros((complex_vertices.shape[0],2), dtype='int64')-1
    labels[:,0] = range(complex_vertices.shape[0])
    for i in range(seeds.shape[0]):
        labels[seeds[i]][1] = i
    # initiate AVLTree for binary search
    tree = FastAVLTree()
    # organisation of the tree will be
    # key: edge length; value: tuple of vertices (source, target)
    # add all neighbours of the voronoi seeds
    for v in seeds:
        add_neighbours(v, 0, graph, labels, tree)
    # Competetive fast marching starting from voronoi seeds
    while tree.count > 0:
        # pop the item with minimum edge length
        min_item = tree.pop_min()
        length = min_item[0]
        source = min_item[1][0]
        target = min_item[1][1]
        #if target no label yet (but source does!), assign label of source
        if labels[target][1] == -1:
            if labels[source][1] == -1:
                sys.exit('Source has no label, something went wrong!')
            else:
                # assign label of source to target
                labels[target][1] = labels[source][1]
        
        # test if labelling is complete
        if any(labels[:,1]==-1):
            # if not, add neighbours of target to tree
            add_neighbours(target, length, graph, labels, tree)
        else:
            break
    
    return labels
    

# main function for looping over subject and hemispheres
def create_mapping((sub, hemi)):

    log_file = '/scr/ilz3/myelinconnect/working_dir/complex_to_simple/fixed_AVL_log_worker_%s.txt'%(str(os.getpid()))
    complex_file = '/scr/ilz3/myelinconnect/struct/surf_%s/orig/mid_surface/%s_%s_mid.vtk'
    simple_file = '/scr/ilz3/myelinconnect/groupavg/indv_space/%s/lowres_%s_d_def.vtk'# version d
    label_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/labels/AVL/%s_%s_highres2lowres_labels.npy'
    surf_label_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/labels/AVL/%s_%s_highres2lowres_labels.vtk'

    # load the meshes
    log(log_file, 'Processing %s %s'%(sub, hemi))
    log(log_file, '...loading data', logtime=False)
    complex_v, complex_f, complex_d = read_vtk(complex_file%(hemi, sub, hemi))
    simple_v, simple_f, simple_d = read_vtk(simple_file%(sub, hemi))

    # find those points on the individuals complex mesh that correspond best
    # to the simplified group mesh in subject space
    log(log_file, '...finding unique voronoi seeds')
    voronoi_seed_idx, inaccuracy  = find_voronoi_seeds(simple_v, complex_v)
    # find coordinates of those points in the highres mesh
    voronoi_seed_coord = complex_v[voronoi_seed_idx]

    # double check differences
    log(log_file, '...checking unique vs nearest mapping')
    dist = np.linalg.norm((voronoi_seed_coord - simple_v), axis=1)
    if ((np.mean(dist)-np.mean(inaccuracy[:,0])>0.1)):
        log(log_file, 'Unique seeds very far from nearest seeds!')
        return dist
        sys.exit("Unique seeds very far from nearest seeds %s %s"%(sub,hemi))

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



if __name__ == "__main__":

    #cachedir = '/scr/ilz3/myelinconnect/working_dir/complex_to_simple/'
    #memory = Memory(cachedir=cachedir)
    
    subjects = pd.read_csv('/scr/ilz3/myelinconnect/subjects.csv')
    subjects=list(subjects['DB'])
    subjects.remove('KSMT')
    subjects.remove('KSYT')
    hemis = ['lh', 'rh']

    Parallel(n_jobs=16)(delayed(create_mapping)(i) 
                               for i in tupler(subjects, hemis))
