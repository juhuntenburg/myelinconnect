
from __future__ import division
import numpy as np
from vtk_rw import read_vtk, write_vtk
from simplification import sample_simple
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

#@memory.cache
def looping(sub, hemi):

    highres_file = '/scr/ilz3/myelinconnect/struct/surf_%s/prep_t1/profiles/%s_%s_mid_proflies.vtk'%(hemi, sub, hemi)
    new_lowres_file = '/scr/ilz3/myelinconnect/final_data_on_surfaces/t1/%s_%s_profiles_lowres.vtk'%(sub, hemi)
    data_file = '/scr/ilz3/myelinconnect/final_data_on_surfaces/t1/%s_%s_profiles_lowres.npy'%(sub, hemi)

    label_file = '/scr/ilz3/myelinconnect/groupavg/indv_space/%s/labelling_%s.txt'%(sub, hemi)
    old_lowres_file = '/scr/ilz3/myelinconnect/groupavg/indv_space/%s/lowres_%s_d_def.vtk'%(sub, hemi)

    # load data
    labels = np.load(label_file)[:,1]
    highres_v, highres_f, highres_d = read_vtk(highres_file)
    lowres_v, lowres_f, lowres_d = read_vtk(old_lowres_file)

    # call to sampling function
    new_lowres_d = sample_simple(highres_d, labels)

    # save lowres vtk and txt
    write_vtk(new_lowres_file, lowres_v, lowres_f, data=new_lowres_d)
    np.save(data_file, new_lowres_d)

    return


if __name__ == "__main__":

    subjects = pd.read_csv('/scr/ilz3/myelinconnect/subjects.csv')
    subjects=list(subjects['DB'])
    subjects.remove('KSMT')

    hemis = ['lh', 'rh']

    cachedir = '/scr/ilz3/working_dir/sample_to_simple/'
    memory = Memory(cachedir=cachedir, mmap_mode='r')

    Parallel(n_jobs=16)(delayed(memory.cache(looping))(i)
                           for i in tupler(subjects, hemis))
