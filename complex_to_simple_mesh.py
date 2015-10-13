from __future__ import division
import numpy as np
import scipy.spatial as spatial
import sys
import time
import pandas as pd
# from https://pypi.python.org/pypi/bintrees/2.0.2
from bintrees import FastAVLTree
# from https://github.com/juhuntenburg/brainsurfacescripts
from vtk_rw import read_vtk, write_vtk
from graphs import graph_from_mesh

'''
This script maps vertices from a simplified mesh to the closest corresponding 
vertices of the original mesh using a KDTree. These vertices are 
then used as seeds for a Voronoi tesselation of the complex mesh which is
implemented as a competitive fast marching in a balanced binary (AVL)tree. 
Thus a mapping is created which associates each vertex of the complex mesh 
with the closest vertex of the simple mesh.
'''

# set inputs 
subjects = pd.read_csv('/scr/ilz3/myelinconnect/subjects.csv')
subjects=list(subjects['DB'])
subjects.remove('KSMT')

complex_file = '/scr/ilz3/myelinconnect/struct/surf_%s/orig/mid_surface/%s_%s_mid.vtk'
simple_file = '/scr/ilz3/myelinconnect/groupavg/indv_space/%s/lowres_%s_d_def.vtk'# version d

label_file = '/scr/ilz3/myelinconnect/groupavg/indv_space/%s/%s_%s_highres2lowres_d_labels.txt'
surf_label_file = '/scr/ilz3/myelinconnect/groupavg/indv_space/%s/%s_%s_highres2lowres_d_labels.vtk'

log_file = '/home/raid3/huntenburg/workspace/myelinconnect/complex_to_simple_log.txt'

# function to find unlabelled neighbours of a node in the graph 
# and add them to the tree correctly
def add_neighbours(node, graph, labels, tree):
    # find direct neighbours of the node
    neighbours = np.array(graph.neighbors(node))
    # check that they don't already have a label
    unlabelled = neighbours[np.where(labels[neighbours][:,1]==-1)[0]]
    # insert source neighbour pair with edge length to tree
    for u in unlabelled:
        tree.insert(graph[node][u]['length'],(node, u))
    return tree



for sub in subjects:
    
    for hemi in ['rh', 'lh']:
        
        with open(log_file, 'a') as f:
            f.write(time.ctime()+'\n')
            f.write('Processing '+sub+' '+hemi+'\n')
            f.write('...loading data\n')

        # load the meshes
        complex_v,complex_f, complex_d = read_vtk(complex_file%(hemi, sub, hemi))
        simple_v, simple_f, simple_d = read_vtk(simple_file%(sub, hemi))
        
        # find those points on the individuals complex mesh that correspond best
        # to the simplified group mesh in subject space
        with open(log_file, 'a') as f:
            f.write(time.ctime()+'\n')
            f.write('...running kdtree\n')
            
        inaccuracy, mapping = spatial.KDTree(complex_v).query(simple_v)
        
        # find coordinates of those points in the highres mesh
        voronoi_seed_idx = mapping
        voronoi_seed_coord = complex_v[mapping]
        
        # double check differences
        with open(log_file, 'a') as f:
            f.write(time.ctime()+'\n')
            f.write('...checking kdtree\n')
            
        dist = np.linalg.norm((voronoi_seed_coord - simple_v), axis=1)
        if any(dist != inaccuracy):
            sys.exit("simple to complex mesh mapping not correct")

        # convert highres mesh into graph containing edge length
        with open(log_file, 'a') as f:
            f.write(time.ctime()+'\n')
            f.write('...making graph\n')
            
        complex_graph = graph_from_mesh(complex_v, complex_f, edge_length=True)

        # make a labelling container to be filled with the search tree
        # first column are the vertex indices of the complex mesh
        # second column are the labels from the simple mesh 
        # (-1 for all but the corresponding points for now)
        with open(log_file, 'a') as f:
            f.write(time.ctime()+'\n')
            f.write('...making labelling\n')
            
        labelling = np.zeros((complex_v.shape[0],2), dtype='int64')-1
        labelling[:,0] = range(complex_v.shape[0])
        
        for i in range(voronoi_seed_idx.shape[0]):
            labelling[voronoi_seed_idx[i]][1] = i
        
        # initiate AVLTree for binary search
        # binary search trees: https://en.wikipedia.org/wiki/Binary_search_tree
        # balanced binary trees: https://en.wikipedia.org/wiki/AVL_tree, 
        # using them as heaps: http://samueldotj.com/blog/?s=binary+tree
        # implementation used: https://pypi.python.org/pypi/bintrees/2.0.2 (cython implementation)
        with open(log_file, 'a') as f:
            f.write(time.ctime()+'\n')
            f.write('...building avltree\n')
            
        tree = FastAVLTree()
        # organisation of the tree will be
        # key: edge length
        # value: tuple of vertices (source, target)


        # find all neighbours of the voronoi seeds
        for v in voronoi_seed_idx:
            add_neighbours(v, complex_graph, labelling, tree)
        
        # Competetive fast marching starting from voronoi seeds
        with open(log_file, 'a') as f:
            f.write(time.ctime()+'\n')
            f.write('...marching\n')
        
        while any(labelling[:,1]==-1):
            while tree.count > 0:
                # pop the item with minimum edge length
                min_item = tree.pop_min()
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
                add_neighbours(target, complex_graph, labelling, tree)
  
        # write out labelling file and surface with labels
        with open(log_file, 'a') as f:
            f.write(time.ctime()+'\n')
            f.write('...saving data\n')
            
        np.savetxt(label_file%(sub, sub, hemi), labelling)
        write_vtk(surf_label_file%(sub, sub, hemi), complex_v, complex_f, data=labelling[:,1, np.newaxis])
        
        with open(log_file, 'a') as f:
            f.write(time.ctime()+'\n')
            f.write('Finished '+sub+' '+hemi+'\n\n')
        