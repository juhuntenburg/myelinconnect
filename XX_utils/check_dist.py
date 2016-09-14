from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from vtk_rw import read_vtk
from plotting import plot_surf_stat_map, crop_img
import matplotlib as mpl
import h5py

lh_mesh_file = '/scr/ilz3/myelinconnect/new_groupavg/surfs/lowres/inflated/lh_lowres_new_infl50.vtk'
lh_sulc_file = '/scr/ilz3/myelinconnect/new_groupavg/surfs/lowres/sulc/lh_sulc.npy'
rh_mesh_file = '/scr/ilz3/myelinconnect/new_groupavg/surfs/lowres/inflated/rh_lowres_new_infl50.vtk'
rh_sulc_file = '/scr/ilz3/myelinconnect/new_groupavg/surfs/lowres/sulc/rh_sulc.npy'
fullmask_file = '/scr/ilz3/myelinconnect/new_groupavg/masks/fullmask_lh_rh_new.npy'
dist_file = '/scr/ilz3/myelinconnect/new_groupavg/corr/both_euclid_dist.hdf5'

def brain_fig(plot_list):
    sns.set_style('white')
    n = len(plot_list)
    rows = int(n/4)
    fig = plt.figure(figsize=(40,rows*5))
    for img in range(n):
        ax = fig.add_subplot(rows,4,img+1)
        plt.imshow(plot_list[img])
        ax.set_axis_off()
    fig.tight_layout()
    fig.subplots_adjust(right=0.65)
    return fig

print 'load mesh'
lh_sulc = np.load(lh_sulc_file)
lv, lf, _ = read_vtk(lh_mesh_file)
rh_sulc = np.load(rh_sulc_file)
rv, rf, _ = read_vtk(rh_mesh_file)
fullmask = np.load(fullmask_file)

print 'load dist'
f = h5py.File(dist_file, 'r')
dist = np.asarray(f['upper'])
full_shape = tuple(f['shape'])
f.close()

print 'full matrix'
full_dist = np.zeros(tuple(full_shape))
full_dist[np.triu_indices_from(full_dist, k=1)] = np.nan_to_num(dist)
del dist
full_dist += full_dist.T


seed = 10000
map = full_dist[seed]
map[fullmask] = 0
vmin=0
vmax=160

cropped_img = []
sns.set_style('white')
for (elev, azim) in [(180, 0), (180, 180)]:
    plot=plot_surf_stat_map(lv, lf, stat_map=map[:lv.shape[0]], bg_map=lh_sulc, bg_on_stat=True, darkness=0.4, 
                        elev=elev,azim=azim, figsize=(10,9) ,threshold=1e-50, cmap='Reds_r',
                        symmetric_cbar=False, vmin=vmin, vmax=vmax)
    cropped_img.append(crop_img(plot))

for (elev, azim) in [(180, 0), (180, 180)]:
    plot=plot_surf_stat_map(rv, rf, stat_map=map[lv.shape[0]:], bg_map=rh_sulc, bg_on_stat=True, darkness=0.4, 
                        elev=elev,azim=azim, figsize=(10,7.5) ,threshold=1e-50, cmap='Reds_r',
                        symmetric_cbar=False, vmin=vmin, vmax=vmax)
    cropped_img.append(crop_img(plot))
    
    
brain_fig(cropped_img)
plt.show()