def create_filter_matrix(motion_params, composite_norm,
                         compcorr_components, art_outliers, global_signal,
                         selector, demean=False):
    """Combine nuisance regressor components into a single file

Parameters
----------
motion_params : parameter file output from realignment
composite_norm : composite norm file from artifact detection
compcorr_components : components from compcor
art_outliers : outlier timepoints from artifact detection
selector : a boolean list corresponding to the files to concatenate together\
[motion_params, composite_norm, compcorr_components, global_signal, art_outliers,\
motion derivatives]
Returns
-------
filter_file : a file with selected parameters concatenated
"""
    import numpy as np
    from scipy.signal import detrend
    import os
    if not len(selector) == 6:
        print "selector is not the right size!"
        return None

    def try_import(fname):
        try:
            a = np.genfromtxt(fname)
            return a
        except:
            return np.array([])

    options = np.array([motion_params, composite_norm,
                        compcorr_components, global_signal, art_outliers])
    selector = np.array(selector)
    fieldnames = ['motion', 'comp_norm', 'compcor', 'global_signal', 'art', 'dmotion']

    splitter = np.vectorize(lambda x: os.path.split(x)[1])
    filenames = [fieldnames[i] for i, val in enumerate(selector) if val]
    filter_file = os.path.abspath("filter_%s.txt" % "_".join(filenames))

    z = None

    for i, opt in enumerate(options[:-1][selector[:-2]]):
    # concatenate all files except art_outliers and motion_derivs
        if i == 0:
            z = try_import(opt)
        else:
            a = try_import(opt)
            if len(a.shape) == 1:
                a = np.array([a]).T
            z = np.hstack((z, a))
    if z is not None and demean:
        z = detrend(z, axis=0, type='constant')

    if selector[-2]:
        #import outlier file
        outliers = try_import(art_outliers)
        if outliers.shape == (): # 1 outlier
            art = np.zeros((z.shape[0], 1))
            art[np.int_(outliers), 0] = 1 # art outputs 0 based indices
            out = np.hstack((z, art))
            
        elif outliers.shape[0] == 0: # empty art file
            out = z
                
        else: # >1 outlier
            art = np.zeros((z.shape[0], outliers.shape[0]))
            for j, t in enumerate(outliers):
                art[np.int_(t), j] = 1 # art outputs 0 based indices
            out = np.hstack((z, art))
    else:
        out = z

    if selector[-1]: # this is the motion_derivs bool
            a = try_import(motion_params)
            temp = np.zeros(a.shape)
            temp[1:, :] = np.diff(a, axis=0)
            if demean:
                out = np.hstack((out, detrend(temp, axis=0, type='constant')))
            else:
                out = np.hstack((out, temp))
            out = np.hstack((out, temp))
    if out is not None:
        np.savetxt(filter_file, out)
        return filter_file
    else:
        filter_file = os.path.abspath("empty_file.txt")
        a = open(filter_file,'w')
        a.close()
        return filter_file

