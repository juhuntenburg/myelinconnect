def avg_correlation(ts_files, thr=None):

    import numpy as np
    import numexpr as ne
    import hcp_corr

    # make empty avg corr matrix
    get_size = np.load(ts_files[0]).shape[0]
    full_shape = (get_size, get_size)
    if np.mod((get_size**2-get_size),2)==0.0:
        avg_corr = np.zeros((get_size**2-get_size)/2)
    else:
        print 'size calculation no zero mod'

    count = 0
    for ts in ts_files:
        # load time series
        print '...load %s'%ts
        rest = np.load(ts)

        # calculate correlations matrix
        print '...corrcoef'
        corr = hcp_corr.corrcoef_upper(rest)
        # corr = np.corrcocoef(rest)
        del rest
        # get upper triangular only
        # corr = corr[np.triu_indices_from(corr, k=1)]

        # threshold / transform
        if thr == None:
            # r-to-z transform and add to avg
            print '...transform'
            avg_corr += ne.evaluate('arctanh(corr)')
        else:
            # threshold and add to avg
            print '...threshold'
            thr = np.percentile(corr, 100-thr)
            avg_corr[np.where(corr>thr)] += 1

        del corr
        count += 1

    # divide by number of sessions included
    print '...divide'
    avg_corr /= count
    # transform back if necessary
    if thr == None:
        print '...back transform'
        avg_corr = np.nan_to_num(ne.evaluate('tanh(avg_corr)'))

    return avg_corr, full_shape
