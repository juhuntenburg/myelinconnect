from vtk_rw import read_vtk
import numpy as np
import gdist


mesh_file="/scr/ilz3/myelinconnect/new_groupavg/surfs/lowres/%s_lowres_new.vtk"
hemis=['rh', 'lh']

for hemi in hemis:
    v, f, _ = read_vtk(mesh_file %hemi)
    
    v = v.astype(np.float64)
    f = f.astype(np.int32)
    dist_matrix=gdist.local_gdist_matrix(v, f)