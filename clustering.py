import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib as plt
from mayavi import mlab
from sklearn.utils.arpack import eigsh
from sklearn.cluster import KMeans


'''
functions
-----------
see also: https://github.com/margulies/topography/blob/master/utils_py/embedding.py
'''

def read_vtk(file):
    # read full file while dropping empty lines 
    vtk_df=pd.read_csv(file, header=None)
    vtk_df=vtk_df.dropna()
    # extract number of vertices and faces
    number_vertices=int(vtk_df[vtk_df[0].str.contains('POINTS')][0].iloc[0].split()[1])
    number_faces=int(vtk_df[vtk_df[0].str.contains('POLYGONS')][0].iloc[0].split()[1])
    # read vertices into df and array
    start_vertices= (vtk_df[vtk_df[0].str.contains('POINTS')].index.tolist()[0])+1
    vertex_df=pd.read_csv(file, skiprows=range(start_vertices), nrows=number_vertices, sep='\s*', header=None)
    if np.array(vertex_df).shape[1]==3:
        vertex_array=np.array(vertex_df)
    # sometimes the vtk format is weird with 9 indices per line, then it has to be reshaped
    elif np.array(vertex_df).shape[1]==9:
        vertex_df=pd.read_csv(file, skiprows=range(start_vertices), nrows=number_vertices/3+1, sep='\s*', header=None)
        vertex_array=np.array(vertex_df.iloc[0:1,0:3])
        vertex_array=np.append(vertex_array, vertex_df.iloc[0:1,3:6], axis=0)
        vertex_array=np.append(vertex_array, vertex_df.iloc[0:1,6:9], axis=0)
        for row in range(1,(number_vertices/3+1)):
            for col in [0,3,6]:
                vertex_array=np.append(vertex_array, array(vertex_df.iloc[row:(row+1),col:(col+3)]),axis=0) 
        # strip rows containing nans
        vertex_array=vertex_array[ ~np.isnan(vertex_array) ].reshape(number_vertices,3)
    else:
        print "vertex indices out of shape"
    vertices = {'val' : vertex_array, 'number' : number_vertices}
    # read faces into df and array
    start_faces= (vtk_df[vtk_df[0].str.contains('POLYGONS')].index.tolist()[0])+1
    face_df=pd.read_csv(file, skiprows=range(start_faces), nrows=number_faces, sep='\s*', header=None)
    face_array=np.array(face_df.iloc[:,1:4])
    faces = {'val' : face_array, 'number' : number_faces}
    # read data into df and array
    start_data= (vtk_df[vtk_df[0].str.contains('POINT_DATA')].index.tolist()[0])+3
    number_data = number_vertices
    data_df=pd.read_csv(file, skiprows=range(start_data), nrows=number_data, sep='\s*', header=None)
    data_array=np.array(data_df)
    data = {'val' : data_array, 'number' : number_data}
    
    return vertices, faces, data


def write_vtk(filename, vertices, faces, data, comment=None):
    # infer number of vertices and faces
    number_vertices=vertices.shape[0]
    number_faces=faces.shape[0]
    number_data=data.shape[0]
    datapoints=data.shape[1]
    # make header and subheader dataframe
    header=['# vtk DataFile Version 3.0',
            '%s'%comment,
            'ASCII',
            'DATASET POLYDATA',
            'POINTS %i float'%number_vertices
             ]
    header_df=pd.DataFrame(header)
    sub_header=['POLYGONS %i %i'%(number_faces, 4*number_faces)]
    sub_header_df=pd.DataFrame(sub_header)    
    sub_header2=['POINT_DATA %i'%(number_data),
                 'SCALARS EmbedVertex float %i'%(datapoints),
                 'LOOKUP_TABLE default']
    sub_header_df2=pd.DataFrame(sub_header2)
    # make dataframe from vertices
    vertex_df=pd.DataFrame(vertices)
    # make dataframe from faces, appending first row of 3's (indicating the polygons are triangles)
    triangles=np.reshape(3*(np.ones(number_faces)), (number_faces,1))
    triangles=triangles.astype(int)
    faces=faces.astype(int)
    faces_df=pd.DataFrame(np.concatenate((triangles,faces),axis=1))
    # make dataframe from data
    data_df=pd.DataFrame(data)
    # write dfs to csv
    header_df.to_csv(filename, header=None, index=False)
    with open(filename, 'a') as f:
        vertex_df.to_csv(f, header=False, index=False, float_format='%.3f', sep=' ')
    with open(filename, 'a') as f:
        sub_header_df.to_csv(f, header=False, index=False)
    with open(filename, 'a') as f:
        faces_df.to_csv(f, header=False, index=False, float_format='%.0f', sep=' ')
    with open(filename, 'a') as f:
        sub_header_df2.to_csv(f, header=False, index=False)
    with open(filename, 'a') as f:
        data_df.to_csv(f, header=False, index=False, float_format='%.16f', sep=' ')


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


'''
running
--------
'''

for sub in ['BP4T','GAET','HJJT','KSYT', 'OL1T', 'PL6T', 'SC1T', 'WSFT']:

    for hemi in ['lh', 'rh']:
        
        print sub, hemi
        
        n_components_embedding=3
        n_components_kmeans=7
        
        func_file='/scr/ilz3/myelinconnect/final_struct_space/rest1_1_meshsmooth_3/%s_%s_mid_simple_0.01_rest_%s_smoothdata.vtk'%(sub, hemi, hemi)
        mask_file='/scr/ilz3/myelinconnect/final_struct_space/medial_wall_mask/%s_%s_mid_simple_0.01.1D.roi'%(sub, hemi)
        embed_file="/scr/ilz3/myelinconnect/final_struct_space/clustering/%s_%s_mid_simple_0.01_rest_%s_smoothdata_embed_%s.csv"%(sub, hemi, hemi, str(n_components_embedding))
        kmeans_file="/scr/ilz3/myelinconnect/final_struct_space/clustering/%s_%s_mid_simple_0.01_rest_%s_smoothdata_embed_%s_kmeans_%s.csv"%(sub, hemi, hemi, str(n_components_embedding), str(n_components_kmeans))
        
        
        print 'reading vtk file'
        v,f,d=read_vtk(func_file)
        func_data=d['val']
        all_vertex=range(len(v['val']))
        
        print 'masking'
        mask=np.loadtxt(mask_file)[:,0]
        masked_func_data=func_data
        masked_func_data=np.delete(masked_func_data, mask, 0)
        cortex=np.delete(all_vertex, mask)
        
        print 'computing correlations'
        corr_data=np.nan_to_num(np.corrcoef(masked_func_data))
        
        print 'running embedding'
        K = (corr_data + 1) / 2.  
        v = np.sqrt(np.sum(K, axis=1)) 
        A = K/(v[:, None] * v[None, :])  
        del K
        A = np.squeeze(A * [A > 0])
        embedding_results = runEmbed(A, n_components_embedding)
        
        embedding_recort=np.zeros((len(all_vertex),embedding_results.shape[1])) 
        for e in range(embedding_results.shape[1]):
            embedding_recort[:,e]=recort(len(all_vertex), embedding_results[:,e], cortex, 0)
        np.savetxt(embed_file, embedding_recort, delimiter=",")
        
        
        print 'running kmeans'
        kmeans_results = runKmeans(embedding_results, n_components_kmeans)
        kmeans_recort = recort(len(all_vertex), kmeans_results, cortex, 1)
        np.savetxt(kmeans_file, kmeans_recort, delimiter=",")
        
        print 'done'


