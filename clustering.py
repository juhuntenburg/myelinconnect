def recort(n_vertices, data, cortex, increase):
    import numpy as np
    d = np.zeros(n_vertices)
    count = 0
    for i in cortex:
        d[i] = data[count] + increase
        count = count +1
    return d

def embedding(upper_corr, full_shape, mask, n_components):
    import numpy as np
    from mapalign import embed

    # reconstruct full matrix
    print '...full matrix'
    full_corr = np.zeros(tuple(full_shape))
    full_corr[np.triu_indices_from(full_corr, k=1)] = np.nan_to_num(upper_corr)
    full_corr += full_corr.T
    all_vertex=range(full_corr.shape[0])

    # apply mask
    print '...mask'
    masked_corr = np.delete(full_corr, mask, 0)
    del full_corr
    masked_corr = np.delete(masked_corr, mask, 1)
    cortex=np.delete(all_vertex, mask)

    # run actual embedding
    print '...embed'
    K = (masked_corr + 1) / 2.
    del masked_corr
    K[np.where(np.eye(K.shape[0])==1)]=1.0
    #v = np.sqrt(np.sum(K, axis=1))
    #A = K/(v[:, None] * v[None, :])
    #del K, v
    #A = np.squeeze(A * [A > 0])
    #embedding_results = runEmbed(A, n_components_embedding)
    embedding_results, embedding_dict = embed.compute_diffusion_map(K, n_components=n_components, overwrite=True)

    # reconstruct masked vertices as zeros
    embedding_recort=np.zeros((len(all_vertex),embedding_results.shape[1]))
    for e in range(embedding_results.shape[1]):
        embedding_recort[:,e]=recort(len(all_vertex), embedding_results[:,e], cortex, 0)

    return embedding_recort, embedding_dict


def t1embedding(upper_corr, full_shape, mask, n_components):
    import numpy as np
    from mapalign import embed

    # reconstruct full matrix
    print '...full matrix'
    full_corr = np.zeros(tuple(full_shape))
    full_corr[np.triu_indices_from(full_corr, k=1)] = np.nan_to_num(upper_corr)
    full_corr += full_corr.T
    all_vertex=range(full_corr.shape[0])

    # apply mask
    print '...mask'
    masked_corr = np.delete(full_corr, mask, 0)
    del full_corr
    masked_corr = np.delete(masked_corr, mask, 1)
    cortex=np.delete(all_vertex, mask)

    # run actual embedding
    print '...embed'
    K=1-(masked_corr/masked_corr.max())
    #K = (masked_corr + 1) / 2.
    del masked_corr
    K[np.where(np.eye(K.shape[0])==1)]=1.0
    #v = np.sqrt(np.sum(K, axis=1))
    #A = K/(v[:, None] * v[None, :])
    #del K, v
    #A = np.squeeze(A * [A > 0])
    #embedding_results = runEmbed(A, n_components_embedding)
    embedding_results, embedding_dict = embed.compute_diffusion_map(K, n_components=n_components, overwrite=True)

    # reconstruct masked vertices as zeros
    embedding_recort=np.zeros((len(all_vertex),embedding_results.shape[1]))
    for e in range(embedding_results.shape[1]):
        embedding_recort[:,e]=recort(len(all_vertex), embedding_results[:,e], cortex, 0)

    return embedding_recort, embedding_dict

def kmeans(embedding,n_components, mask):
    import numpy as np
    from sklearn.cluster import KMeans
    
    all_vertex=range(embedding.shape[0])
    masked_embedding = np.delete(embedding, mask, 0)
    cortex=np.delete(all_vertex, mask)
    
    est = KMeans(n_clusters=n_components, n_jobs=-2, init='k-means++', n_init=300)
    est.fit_transform(masked_embedding)
    labels = est.labels_
    kmeans_results = labels.astype(np.float)
    kmeans_recort = recort(len(all_vertex), kmeans_results, cortex, 1)
    return kmeans_recort


def subcluster(kmeans, triangles):
    
    import numpy as np
    # make a dictionary for kmeans clusters and subclusters
    subclust={}
    # loop through all kmeans clusters (and the mask cluster with value zero)
    for c in range(int(kmeans.max()+1)):
        # add dic entry
        subclust['k'+str(c)]=[]
        # extract all nodes of the cluster
        clust=list(np.where(kmeans==c)[0])
        # while not all nodes have been removed from the cluster
        while len(clust)>0:
            #start at currently first node in cluster
            neighbours=[clust[0]]
            # go through growing list of neighbours in the subcluster
            for i in neighbours:
                #find all triangles that contain current
                for j in range(triangles.shape[0]):
                    if i in triangles[j]:
                        # add all nodes of in this triangle to the neighbours list
                        n=list(triangles[j])
                        # but only if they aren't already in the list and if they are in clust
                        [neighbours.append(x) for x in n if x in clust and x not in neighbours]
                        # remove assigned nodes from the cluster list
                        [clust.remove(x) for x in n if x in clust]
            # when no new neighbours can be found, add subcluster to subcluster list 
            # and start new subcluster from first node in remaining cluster list
            subclust['k'+str(c)].append(neighbours)
    
    # make array with original kmeans clusters and subclusters        
    subclust_full = np.zeros((kmeans.shape[0], int(kmeans.max()+1)))
    count = 0
    for c in range(len(subclust.keys())):
        for i in range(len(subclust['k'+str(c)])):
            for j in subclust['k'+str(c)][i]:
                subclust_full[j][c] = i+1
    #subclust_arr=np.hstack((np.reshape(kmeans, (kmeans.shape[0],1)), subclust_full))
    return subclust_full



def adjacent_cluster(subcluster, edges):
    
    
    
