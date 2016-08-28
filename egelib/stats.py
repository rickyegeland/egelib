import numpy as np

def printstats(x, label, precision=3):
    """Print basic statistics about a 1D array"""
    fmt = "N=%%i min=%%0.%(p)if max=%%0.%(p)if median=%%0.%(p)if mean=%%0.%(p)if std=%%0.%(p)if" % dict(p=precision)
    print label, fmt % \
    (x.size, x.min(), x.max(), np.median(x), x.mean(), x.std())

def chi2_gof(x, model, e_x, n_params=0):
    """Chi 2 goodness-of-fit test"""
    chi2 = ((x - model)**2 / e_x**2).sum()
    n_obs = x.size
    dof = n_obs - n_params - 1
    return chi2, chi2/dof

def bin_equalN(x, Ninbin, sorted=False):
    if not sorted:
        x = np.sort(x)
    N = x.size
    ixbins = np.arange(Ninbin, N, Ninbin, dtype='i')
    bins = np.zeros(ixbins.size + 1)
    bins[0] = x[0]
    for i in range(ixbins.size):
        bins[i+1] = (x[ixbins[i]] + x[ixbins[i] - 1])/2
    return bins

def mad(x):
    """Median absolute deviation"""
    return np.median(np.abs(x - np.median(x)))

def mad_sigma(x):
    """Median absolute deviation scaled to 1-sigma in case of Gaussian distribution """
    return 1.483 * mad(x)

def mad_outliers(x, bool=False, Nsigma=4):
    """Identify outliers using the median absolute deviation

    Nsigma is the sigma-threshold for outliers in the case of a
    Gaussian distribution.  As MAD is designed for non-Gaussian
    distributions, specifying the threshold in this way is merely a
    convenience.
    """
    med = np.median(x)
    mad_std = mad_sigma(x)
    out = np.abs(x - med) > Nsigma * mad_std # bool array
    if bool is False:
        out = np.arange(x.size)[out] # index array
    return out
