from __future__ import division
import numpy as np
from vtk_rw import read_vtk, write_vtk
from simplification import sample_simple
from joblib import Memory, Parallel, delayed
import os
import pandas as pd
import scipy as sp

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

def tupler(subjects, hemis):
    for s in subjects:
        for h in hemis:
            yield (s, h)

#@memory.cache
def looping((sub, hemi)):

    highres_file = '/scr/ilz3/myelinconnect/struct/surf_%s/prep_t1/profiles/%s_%s_mid_proflies.vtk'%(hemi, sub, hemi)
    label_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/labels/%s_%s_highres2lowres_labels.npy'%(sub, hemi)
    old_lowres_file = '/scr/ilz3/myelinconnect/groupavg/indv_space/%s/lowres_%s_d_def.vtk'%(sub, hemi)

    
    new_lowres_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/t1/raw/%s_%s_profiles_raw.vtk'%(sub, hemi)
    data_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/t1/raw/%s_%s_profiles_raw.npy'%(sub, hemi)
    cheb_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/t1/raw/%s_%s_coeff_raw.npy'%(sub, hemi)
    cheb_vtk_file = '/scr/ilz3/myelinconnect/all_data_on_simple_surf/t1/raw/%s_%s_coeff_raw.vtk'%(sub, hemi)


    # load data
    labels = np.load(label_file)[:,1]
    highres_v, highres_f, highres_d = read_vtk(highres_file)
    lowres_v, lowres_f, lowres_d = read_vtk(old_lowres_file)

    # call to sampling function
    new_lowres_d = sample_simple(highres_d, labels)

    # save lowres vtk and txt
    write_vtk(new_lowres_file, lowres_v, lowres_f, data=new_lowres_d)
    np.save(data_file, new_lowres_d)
    
    # calculate chebychev coefficients
    t1_3_7 = new_lowres_d[:,3:8]
    coeff, poly = chebapprox(t1_3_7, degree=4)
    np.save(cheb_file, coeff)
    write_vtk(cheb_vtk_file, lowres_v, lowres_f, data=coeff)
    
    if os.path.isfile(data_file):
        print sub+' '+hemi+' finished'

    return


if __name__ == "__main__":

    subjects = pd.read_csv('/scr/ilz3/myelinconnect/subjects.csv')
    subjects=list(subjects['DB'])
    subjects.remove('KSMT')

    hemis = ['rh', 'lh']

    #cachedir = '/scr/ilz3/working_dir/sample_to_simple/'
    #memory = Memory(cachedir=cachedir, mmap_mode='r')

    #Parallel(n_jobs=8)(delayed(looping)(i)
#                           for i in tupler(subjects, hemis))
    
    for sub in subjects:
        for hemi in hemis:
            print sub, hemi
            looping((sub, hemi))
