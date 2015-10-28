import numpy as np
import gdist
from vtk_rw import read_vtk
import time

sub = 'BP4T'
hemi = 'lh'

complex_file = '/scr/ilz3/myelinconnect/struct/surf_%s/orig/mid_surface/%s_%s_mid.vtk'%(hemi, sub, hemi)
simple_file = '/scr/ilz3/myelinconnect/groupavg/indv_space/%s/lowres_%s_d_def.vtk'%(sub, hemi)# version d

complex_v, complex_f, complex_d = read_vtk(complex_file)
complex_vertices = complex_v.astype(np.float64)
complex_faces = complex_f.astype(np.int32)

simple_v, simple_f, simple_d = read_vtk(simple_file)
simple_vertices = simple_v.astype(np.float64)
simple_faces = simple_f.astype(np.int32)

complex_radius = 2 

print time.ctime()
complex_matrix=gdist.local_gdist_matrix(complex_vertices, complex_faces, max_distance=complex_radius)

for sv in range(len(simple_v.shape[0])):
    dist = 1000000
    for cv in range(len(complex_v.shape[0])):
        if sv in complex_matrix[:,cv].indices:
            dist_now = complex_matrix[:,cv][sv].data
            if dist_now < dist:
                dist = dist_now
                


quatsch, need to tunr around
             
#    inradius=list(complex_matrix[:,v].indices)
    # find




