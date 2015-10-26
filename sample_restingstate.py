from __future__ import division
import numpy as np
from vtk_rw import read_vtk, write_vtk
from simplification import sample_simple
from joblib import Memory, Parallel, delayed
import nibabel as nb

def tupler(subjects, hemis, rests):
    for s in subjects:
        for h in hemis:
            for r in rests:
                yield (s, h, r)

#@memory.cache
def looping((sub, hemi, rest)):

    label_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/labels/%s_%s_highres2lowres_labels.npy'%(sub, hemi)
    rest_file = '/scr/ilz3/myelinconnect/resting/final/%s_rest%s_denoised.nii.gz'%(sub, rest)
    highres_file = '/scr/ilz3/myelinconnect/struct/surf_%s/orig2func/%s_%s_mid_groupavgsurf.vtk'%(hemi, sub, hemi)
        
    data_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/rest/%s_%s_rest%.npy'%(sub, hemi, rest)

    # load data
    labels = np.load(label_file)[:,1]
    highres_v, highres_f, highres_d = read_vtk(highres_file)
    
    img = nb.load(rest_file)
    affine = img.get_affine()
    rest = img.get_data()
    
    # for each vertex in the highres mesh find voxel it maps to
    dim = -(np.round([affine[0,0], affine[1,1], affine[2,2]], 1))
    idx = np.asarray(np.round(highres_v/dim), dtype='int64')
    rest_highres = rest[idx[:,0],idx[:,1],idx[:,2]]
    
    # average across highres vertices that map to the same lowres vertex
    rest_lowres = sample_simple(rest_highres, labels)

    # save data
    np.save(data_file, rest_lowres)

    return data_file


if __name__ == "__main__":

    subjects = pd.read_csv('/scr/ilz3/myelinconnect/subjects.csv')
    subjects=list(subjects['DB'])
    subjects.remove('KSMT')

    hemis = ['lh', 'rh']
    rests = ['1_1', '1_2', '2_1', '2_2']

    cachedir = '/scr/ilz3/working_dir/sample_to_simple/'
    memory = Memory(cachedir=cachedir, mmap_mode='r')

    Parallel(n_jobs=16)(delayed(memory.cache(looping))(i)
                           for i in tupler(subjects, hemis, rests))
