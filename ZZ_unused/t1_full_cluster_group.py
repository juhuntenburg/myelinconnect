from __future__ import division
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from mayavi import mlab
import seaborn as sns
from vtk_rw import read_vtk, write_vtk
import operator

hemis=['rh']
n_embed = 10
n_kmeans = range(2,21)
smooths = ['smooth_2','smooth_3']


def make_cmap(c):
    cmap = np.asarray(sns.color_palette('cubehelix', c-1))
    cmap = np.concatenate((np.array([[0.4,0.4,0.4]]), cmap), axis=0)
    cmap = np.concatenate((cmap, np.ones((c,1))), axis=1)
    cmap_seaborn = [tuple(cmap[i]) for i in range(len(cmap))]

    cmap_255=np.zeros_like(cmap)
    for row in range(cmap.shape[0]):
        cmap_255[row]=[np.floor(i * 255) for i in cmap[row]]
    cmap_255=cmap_255.astype(int)
    
    return cmap_seaborn, cmap_255


def chebapprox(profiles, degree):
    profiles=np.array(profiles)
    cheb_coeffs=np.zeros((profiles.shape[0],degree+1))
    cheb_polynoms=np.zeros((profiles.shape[0],profiles.shape[1]))
    for c in range(profiles.shape[0]):
        x=np.array(range(profiles.shape[1]))
        y=profiles[c]
        cheb_coeffs[c]=np.polynomial.chebyshev.chebfit(x, y, degree)
        cheb_polynoms[c]=np.polynomial.chebyshev.chebval(x, cheb_coeffs[c])
    return cheb_coeffs, cheb_polynoms



for hemi in hemis:
    for smooth in smooths:
        for nk in n_kmeans:

            pal, pal_255 = make_cmap(nk+1)
            
            mesh_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/surfs/lowres_%s_d.vtk'%hemi
            mask_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/masks/%s_mask.1D.roi'%hemi
            embed_file='/scr/ilz3/myelinconnect/all_data_on_simple_surf/clust/%s/%s_embed_%s.npy'%(smooth, hemi, str(n_embed))
            kmeans_file='/scr/ilz3/myelinconnect/all_data_on_simple_surf/clust/%s/%s_embed_%s_kmeans_%s.npy'%(smooth, hemi, str(n_embed), str(nk))
            t1_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/t1/avg_%s_profiles.npy'%(hemi)
            t1clust_file='/scr/ilz3/myelinconnect/pics/new_clustering/t1_in_cluster/%s_%s_kmeans_%s_%s.png'


            v,f,d = read_vtk(mesh_file)
            x=v[:,0]
            y=v[:,1]
            z=v[:,2]
            triangles=f


            mask = np.loadtxt(mask_file)[:,0]
            embed=np.load(embed_file)
            clust=np.load(kmeans_file)
            t1=np.load(t1_file)

            
            # make a list of dictionaries for each cluster k0, k1, ...(0=mask to max kmeans) 
            t1_clust={}
            for c in range(int(clust.max()+1)):
                t1_clust['k'+str(c)]=[]
            
            # write all t1 profiles in one cluster into the list of its dictionary
            for i in range(len(t1)):
                k=int(clust[i])
                t1_clust['k'+str(k)].append(t1[i])
                
            # make other dictionaries with 3 to 7 intracortical profiles and averages, thresholded  1400 to 2500 mean
            t1_clust_37={}
            t1_clust_37_avg={}
            t1_clust_37_coeff_0={}
            t1_clust_37_coeff_1={}
            t1_clust_37_coeff_2={}
            t1_clust_37_coeff_3={}
            for c in range(int(clust.max()+1)):
                t1_clust_37['k'+str(c)]=[pro[3:8] for pro in t1_clust['k'+str(c)]
                                        if np.mean(pro[3:8]) < 2500
                                        and np.mean(pro[3:8]) > 1400]
                t1_clust_37_avg['k'+str(c)]=[np.mean(pro_37) for pro_37 in t1_clust_37['k'+str(c)]]
                
                cheb_coeffs, cheb_polynoms = chebapprox(t1_clust_37['k'+str(c)], 10)
                
                t1_clust_37_coeff_0['k'+str(c)]=[cheb_coeffs[j][0] for j in range(len(t1_clust_37['k'+str(c)]))]
                t1_clust_37_coeff_1['k'+str(c)]=[cheb_coeffs[j][1] for j in range(len(t1_clust_37['k'+str(c)]))]
                t1_clust_37_coeff_2['k'+str(c)]=[cheb_coeffs[j][2] for j in range(len(t1_clust_37['k'+str(c)]))]
                t1_clust_37_coeff_3['k'+str(c)]=[cheb_coeffs[j][3] for j in range(len(t1_clust_37['k'+str(c)]))]


            T_dicts = [t1_clust_37_avg, t1_clust_37_coeff_1, t1_clust_37_coeff_2, t1_clust_37_coeff_3]
            T_names = ['t1_avg', 't1_coeff_1', 't1_coeff_2', 't1_coeff_3']

            for i in range(len(T_dicts)):
                
                T = T_dicts[i]
                
                t_array=np.zeros((int(clust.max()),int(clust.max())))
                p_array=np.zeros((int(clust.max()),int(clust.max())))
                col2=[]
                for c1 in range(int(clust.max())):
                    for c2 in range(int(clust.max())):
                        if c2>=c1:
                            a=T['k'+str(c1+1)]
                            b=T['k'+str(c2+1)]
                            t,p=stats.ttest_ind(a, b)
                            t_array[c1][c2]=t
                            p_array[c1][c2]=p
                    col2.append(c1+1)
                t_df=pd.DataFrame(t_array, columns=col2, index=col2)
                p_df=pd.DataFrame(p_array, columns=col2, index=col2)


                means = {}
                for c in range(int(clust.max())):
                    means['k'+str(c+1)] = np.mean(T['k'+str(c+1)])
                sorted_means = sorted(means.items(), key=operator.itemgetter(1))
                sorted_pal = [pal[int(sorted_means[p][0][1:])] for p in range(len(sorted_means))] 
                
                plot_list=[]
                cluster_list=[]
                for m in range(len(sorted_means)):
                    plot_list+=T[sorted_means[m][0]]
                    cluster_list+=len(T[sorted_means[m][0]])*[sorted_means[m][0]]
                    
                plot_df=pd.DataFrame(columns=['t1', 'cluster'])
                plot_df['t1']=plot_list
                plot_df['cluster']=cluster_list


                sns.set_context('notebook', font_scale=1.8)
                fig = plt.figure(figsize=(15,8))
                sns.violinplot(x='cluster', y='t1',data=plot_df, palette=sorted_pal, inner=None, saturation=1)
                sns.axlabel('',T_names[i], fontsize=22)
                fig.savefig(t1clust_file%(smooth, hemi, str(nk), T_names[i]))
