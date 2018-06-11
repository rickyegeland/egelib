from __future__ import print_function
import numpy as np
import scipy.stats

def printstats(x, label, precision=3):
    """Print basic statistics about a 1D array"""
    fmt = "N=%%i min=%%0.%(p)if max=%%0.%(p)if median=%%0.%(p)if mean=%%0.%(p)if std=%%0.%(p)if" % dict(p=precision)
    print(label, fmt % \
    (x.size, x.min(), x.max(), np.median(x), x.mean(), x.std()))

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

def ols_alpha(beta, x_mean, y_mean):
    # OLS-bisector: Isobe et al. 1990
    return y_mean - beta * x_mean

def ols_bisector_beta(beta1, beta2):
    beta3 = (beta1*beta2 - 1. + np.sqrt((1. + beta1**2)*(1. + beta2**2)))/(beta1 + beta2)
    return beta3

def ols_bisector_params(beta1, beta2, x_mean, y_mean):
    beta3 = ols_bisector_beta(beta1, beta2)
    alpha3 = ols_alpha(beta3, x_mean, y_mean)
    return [beta3, alpha3]

def ols_bisector(x, y):
    params1 = scipy.stats.linregress(x, y)
    params2 = scipy.stats.linregress(y, x)
    params2i = [1./params2[0], -params2[1]/params2[0]]
    return ols_bisector_params(params1[0], params2i[0], np.mean(x), np.mean(y))

def ols_bisector_werrs_beta(x, y, e_x, e_y):
    # Feigelson & Babu 1992 equations 1-6, but see erratum
    e_ratio = (e_y/e_x)**2
    ((Sxx, Sxy), (Syx, Syy)) = np.cov(x, y, ddof=0)
    S_diff = (Syy - e_ratio*Sxx)
    beta = (S_diff + np.sqrt(S_diff**2 + 4*e_ratio*Sxy**2))/(2*Sxy)
    n = x.size
    R = np.sum((y - y.mean() - beta*(x - x.mean()))**2)/(n-2)
    var_B = (beta/Sxy)**2 * (R*Sxy/beta + R**2*(n-2)*(e_ratio+beta**2/(n-1))/ \
                             ((n-1)*(e_ratio + beta**2)**2))
    e_beta = np.sqrt(var_B)
    return beta, e_beta

def ols_bisector_werrs(x, y, e_x, e_y):
    beta, e_beta = ols_werrs_beta(x, y, e_x, e_y)
    alpha = ols_alpha(beta, np.mean(x), np.mean(y))
    return ((beta, alpha), e_beta)
