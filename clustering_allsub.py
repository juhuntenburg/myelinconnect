import numpy as np
import scipy.stats as stats
#from sklearn.utils.arpack import eigsh
from sklearn.cluster import KMeans
from mapalign import compute_diffusion_map
import h5py


'''
inputs
-------
'''
hemis = ['rh']

#
kclusts = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
#kclusts = [7,17]
embed = 3
smooth='raw'

mask_file='/scr/ilz3/myelinconnect/all_data_on_simple_surf/masks/%s_mask.1D.roi'

corr_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/corr/%s_%s_avg_corr.hdf5'
embed_file="/scr/ilz3/myelinconnect/all_data_on_simple_surf/clust/%s/avg/%s_embed_%s.csv"
kmeans_file="/scr/ilz3/myelinconnect/all_data_on_simple_surf/clust/%s/avg/%s_kmeans_%s_embed_%s.csv"
#corr_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/corr/%s_%s_thr_per_session_corr.hdf5'
#embed_file="/scr/ilz3/myelinconnect/all_data_on_simple_surf/clust/%s/thr/%s_embed_%s.csv"
#kmeans_file="/scr/ilz3/myelinconnect/all_data_on_simple_surf/clust/%s/thr/%s_kmeans_%s_embed_%s.csv"


'''
functions
-----------

'''

#def runEmbed(data, n_components):
#    lambdas, vectors = eigsh(data, k=n_components)   
#    lambdas = lambdas[::-1]  
#    vectors = vectors[:, ::-1]  
#    psi = vectors/vectors[:, 0][:, None]  
#    lambdas = lambdas[1:] / (1 - lambdas[1:])  
#    embedding = psi[:, 1:(n_components + 1)] * lambdas[:n_components][None, :]  
    #embedding_sorted = np.argsort(embedding[:], axis=1)
#    return embedding

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
        f = h5py.File(corr_file%(hemi, smooth), 'r')
        corr = np.zeros((f['shape']))
        corr[np.triu_indices_from(corr, k=1)] = np.nan_to_num(np.asarray(f['upper']))
        corr += corr.T
        all_vertex=range(corr.shape[0])
        
        print 'masking'
        mask = np.loadtxt(mask_file%(hemi))[:,0]
        masked_corr = np.delete(corr, mask, 0)
        masked_corr = np.delete(masked_corr, mask, 1)
        del corr
        cortex=np.delete(all_vertex, mask)
        
        
        try:
            np.loadtxt(embed_file%(smooth, hemi, str(n_components_embedding)))
            
        except IOError: 
            
            print 'running embedding'
            K = (masked_corr + 1) / 2.  
            del masked_corr
            v = np.sqrt(np.sum(K, axis=1)) 
            A = K/(v[:, None] * v[None, :])  
            del K
            del v
            A = np.squeeze(A * [A > 0])
            #embedding_results = runEmbed(A, n_components_embedding)
            embedding_results = compute_diffusion_map(A, n_components=10)
        
            embedding_recort=np.zeros((len(all_vertex),embedding_results.shape[1])) 
            for e in range(embedding_results.shape[1]):
                embedding_recort[:,e]=recort(len(all_vertex), 
                                             embedding_results[:,e], cortex, 0)
            np.savetxt(embed_file%(smooth, hemi, str(n_components_embedding)), 
                       embedding_recort, delimiter=",")
        
        print 'running kmeans'
        kmeans_results = runKmeans(embedding_results, n_components_kmeans)
        kmeans_recort = recort(len(all_vertex), kmeans_results, cortex, 1)
        np.savetxt(kmeans_file%(smooth, hemi, str(n_components_kmeans), 
                                str(n_components_embedding)), 
                   kmeans_recort, delimiter=",")
      
        
        print 'done'