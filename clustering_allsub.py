import numpy as np
import scipy.stats as stats
from sklearn.utils.arpack import eigsh
from sklearn.cluster import KMeans


'''
inputs
-------
'''
hemis = ['rh']

kclusts = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
embed = 3
          
#corr_file='/scr/ilz3/myelinconnect/all_data_on_simple_surf/corr/%s_allsub_avg_corr.npy'
corr_file='/scr/ilz3/myelinconnect/all_data_on_simple_surf/corr/%s_allsub_avg_corr_thr.npy'
mask_file='/scr/ilz3/myelinconnect/all_data_on_simple_surf/surfs/%s_mask.1D.roi'
embed_file="/scr/ilz3/myelinconnect/all_data_on_simple_surf/clust/thr/%s_embed_%s.csv"
kmeans_file="/scr/ilz3/myelinconnect/all_data_on_simple_surf/clust/thr/%s_kmeans_%s_embed_%s.csv"


'''
functions
-----------

'''

def runEmbed(data, n_components):
    lambdas, vectors = eigsh(data, k=n_components)   
    lambdas = lambdas[::-1]  
    vectors = vectors[:, ::-1]  
    psi = vectors/vectors[:, 0][:, None]  
    lambdas = lambdas[1:] / (1 - lambdas[1:])  
    embedding = psi[:, 1:(n_components + 1)] * lambdas[:n_components][None, :]  
    #embedding_sorted = np.argsort(embedding[:], axis=1)
    return embedding

def runKmeans(embedding, n_components):
    est = KMeans(n_clusters=n_components, n_jobs=-2, init='k-means++', n_init=300)
    est.fit_transform(embedding)
    labels = est.labels_
    data = labels.astype(np.float)
    return data

def recort(n_vertices, data, cortex, increase):
    d = np.zeros(n_vertices)
    count = 0
    for i in cortex:
        d[i] = data[count] + increase
        count = count +1
    return d

# def subcluster(kmeans, triangles, n_vertices):
#     # make a dictionary for kmeans clusters and subclusters
#     clust_subclust={}
#     # loop through all kmeans clusters (and the mask cluster with value zero)
#     for c in range(int(kmeans.max()+1)):
#         # add dic entry
#         clust_subclust['k'+str(c)]=[]
#         # extract all nodes of the cluster
#         clust=list(np.where(kmeans==c)[0])
#         # while not all nodes have been removed from the cluster
#         while len(clust)>0:
#             #start at currently first node in cluster
#             neighbours=[clust[0]]
#             # go through growing list of neighbours in the subcluster
#             for i in neighbours:
#                 #find all triangles that contain current
#                 for j in range(triangles.shape[0]):
#                     if i in triangles[j]:
#                         # add all nodes of in this triangle to the neighbours list
#                         n=list(triangles[j])
#                         # but only if they aren't already in the list and if they are in clust
#                         [neighbours.append(x) for x in n if x in clust and x not in neighbours]
#                         # remove assigned nodes from the cluster list
#                         [clust.remove(x) for x in n if x in clust]
#             # when no new neighbours can be found, add subcluster to subcluster list 
#             # and start new subcluster from first node in remaining cluster list
#             clust_subclust['k'+str(c)].append(neighbours)
#     
#     # make array with original kmeans clusters and subclusters        
#     subclust_full = np.zeros((n_vertices, int(kmeans.max()+1)))
#     count = 0
#     for c in range(len(clust_subclust.keys())):
#         for i in range(len(clust_subclust['k'+str(c)])):
#             for j in clust_subclust['k'+str(c)][i]:
#                 subclust_full[j][c] = i+1
#     subclust_arr=np.hstack((np.reshape(kmeans, (kmeans.shape[0],1)), subclust_full))
#     
#     return subclust_arr


'''
running
--------
'''

for hemi in hemis:
        
    for kclust in kclusts:
    
        print hemi, kclust
        
        n_components_embedding=embed
        n_components_kmeans=kclust
         
        print 'reading corr data'
        corr = np.nan_to_num(np.load(corr_file%(hemi)))
        all_vertex=range(corr.shape[0])
        
        print 'masking'
        mask = np.loadtxt(mask_file%(hemi))[:,0]
        masked_corr = np.delete(corr, mask, 0)
        masked_corr = np.delete(masked_corr, mask, 1)
        del corr
        cortex=np.delete(all_vertex, mask)
        
        print 'running embedding'
        K = (masked_corr + 1) / 2.  
        v = np.sqrt(np.sum(K, axis=1)) 
        A = K/(v[:, None] * v[None, :])  
        del K
        A = np.squeeze(A * [A > 0])
        embedding_results = runEmbed(A, n_components_embedding)
        
        embedding_recort=np.zeros((len(all_vertex),embedding_results.shape[1])) 
        for e in range(embedding_results.shape[1]):
            embedding_recort[:,e]=recort(len(all_vertex), 
                                         embedding_results[:,e], cortex, 0)
        np.savetxt(embed_file%(hemi, str(n_components_embedding)), 
                   embedding_recort, delimiter=",")
        
        print 'running kmeans'
        kmeans_results = runKmeans(embedding_results, n_components_kmeans)
        kmeans_recort = recort(len(all_vertex), kmeans_results, cortex, 1)
        np.savetxt(kmeans_file%(hemi, str(n_components_kmeans), 
                                str(n_components_embedding)), 
                   kmeans_recort, delimiter=",")
        
        #print 'subclustering'
        #subclust_arr=subcluster(kmeans_recort, face['val'], n_vertices)
        #np.savetxt(subclust_file, subclust_arr, delimiter=",")        
        
        print 'done'