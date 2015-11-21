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


#def runEmbed(data, n_components):
#    from sklearn.utils.arpack import eigsh
#    lambdas, vectors = eigsh(data, k=n_components)
#    lambdas = lambdas[::-1]
#    vectors = vectors[:, ::-1]
#    psi = vectors/vectors[:, 0][:, None]
#    lambdas = lambdas[1:] / (1 - lambdas[1:])
#    embedding = psi[:, 1:(n_components + 1)] * lambdas[:n_components][None, :]
    #embedding_sorted = np.argsort(embedding[:], axis=1)
#    return embedding
