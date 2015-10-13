
from __future__ import division
import numpy as np
from vtk_rw import read_vtk, write_vtk
from joblib import Memory, Parallel, delayed

'''
Takes a highres to lowres mesh labelling, finds all vertices of the 
highres mesh that are assigned to a given lowres vertex, samples the data
from these vertices and assigns their mean to the lowres vertex
'''

def tupler(subjects, hemis):
    for s in subjects:
        for h in hemis:
            yield (s, h)

@memory.cache
def sample_to_simple_surf(sub, hemi):

    highres_file = '/scr/ilz3/myelinconnect/struct/surf_%s/prep_t1/profiles/%s_%s_mid_proflies.vtk'%(hemi, sub, hemi)
    #highres_file = avg
    new_lowres_file = '/scr/ilz3/myelinconnect/final_data_on_surfaces/t1/%s_%s_profiles_lowres.vtk'%(sub, hemi)
    # new_lowres_file = avg
    data_file = '/scr/ilz3/myelinconnect/final_data_on_surfaces/t1/%s_%s_profiles_lowres.npy'%(sub, hemi)
    # data file = avg

    label_file = '/scr/ilz3/myelinconnect/groupavg/indv_space/%s/labelling_%s.txt'%(sub, hemi)
    old_lowres_file = '/scr/ilz3/myelinconnect/groupavg/indv_space/%s/lowres_%s_d_def.vtk'%(sub, hemi)

    # load data
    labels = np.loadtxt(label_file)
    highres_v, highres_f, highres_data = read_vtk(highres_file)
    lowres_v, lowres_f, lowres_d = read_vtk(old_lowres_file)

    # create new empty lowres data array
    lowres_data = np.empty((int(labels[:,1].max()+1), highres_data.shape[1]))

    # find all vertices on highres and mean
    for l in range(int(labels[:,1 ].max())):
        patch = np.where(labels[:,1]==l)[0]
        patch_data = highres_data[patch]
        patch_mean = np.mean(patch_data, axis=0)
        lowres_data[l] = patch_mean

    # save lowres vtk and txt
    write_vtk(new_lowres_file, lowres_v, lowres_f, data=lowres_data)
    np.save(data_file, lowres_data)

    return


if __name__ == "__main__":

    subjects = pd.read_csv('/scr/ilz3/myelinconnect/subjects.csv')
    subjects=list(subjects['DB'])
    subjects.remove('KSMT')

    hemis = ['lh', 'rh']

    cachedir = '/scr/ilz3/working_dir/sample_to_simple/'
    memory = Memory(cachedir=cachedir, mmap_mode='r')

    Parallel(n_jobs=4)(delayed(sample_to_simple)(i)
                           for i in tupler(subjects, hemis))
