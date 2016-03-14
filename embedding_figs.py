from plotting import plot_surf_stat_map, crop_img
from vtk_rw import read_vtk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

for method in ['markov', 'cauchy']:

    for hemi in ['rh', 'lh']:
        mesh_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/surfs/lowres_%s_d.vtk' % hemi
        sulc_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/surfs/lowres_%s_d_sulc.npy' %  hemi
        #embedding_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/embed/connectivity/%s_smooth_3_embed_10.npy' % hemi
        #fig_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/embed/figs/connectivity_%s_comp%s.png'
        #embedding_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/embed/profiles/%s_smooth_3_embedding_10_%s.npy' % (hemi, method)
        #fig_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/embed/figs/profiles_%s_%s_comp%s.png'
        embedding_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/embed/coefficients/%s_smooth_3_embedding_10_%s.npy' % (hemi, method)
        fig_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/embed/figs/coefficients_%s_%s_comp%s.png'
            
        v, f, d = read_vtk(mesh_file)
        data = np.load(embedding_file)
        sulc = np.load(sulc_file)
    
        for e in range(10):
            
            print method, hemi, e
            
            if hemi == 'lh':
                sns.set_style('white')
                lat=plot_surf_stat_map(v, f, stat_map=data[:,e], bg_map=sulc, bg_on_stat=True,
                                    elev=180,azim=180, figsize=(14,10), darkness=0.3)
    
                sns.set_style('white')
                med=plot_surf_stat_map(v, f, stat_map=data[:,e], bg_map=sulc, bg_on_stat=True,
                        elev=180,azim=0, figsize=(14,10), darkness=0.3)
                
                
            elif hemi == 'rh':
                sns.set_style('white')
                lat=plot_surf_stat_map(v, f, stat_map=data[:,e], bg_map=sulc, bg_on_stat=True,
                                    elev=180,azim=0, figsize=(11,10), darkness=0.4)
        
                sns.set_style('white')
                med=plot_surf_stat_map(v, f, stat_map=data[:,e], bg_map=sulc, bg_on_stat=True,
                        elev=180,azim=180, figsize=(11,10), darkness=0.4)
    
        
            lat_crop=crop_img(lat)
            med_crop=crop_img(med)
                    
            fig=plt.figure()
            fig.set_size_inches(8, 4)
            #ax1 = plt.subplot2grid((4,60), (0,0),  colspan = 26, rowspan =4)
            ax1 = fig.add_subplot(121)
            plt.imshow(lat_crop)
            ax1.set_axis_off()
            #ax2 = plt.subplot2grid((4,60), (0,28), colspan = 26, rowspan =4)
            ax2 = fig.add_subplot(122)
            plt.imshow(med_crop)
            ax2.set_axis_off()
            
            fig.savefig(fig_file % (method, hemi, str(e+1)))
            plt.close(fig)