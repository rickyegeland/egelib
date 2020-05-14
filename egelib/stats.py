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

def ols_xy(x, y):
    # NOTE: this appears to be mislabeled in Isobe 1990 as OLS(X|Y) in Isobe, but is actually OLS(Y|X)
    # Gives same result, except for e_beta1, as scipy.stats.linregress
    # Produces a smaller residual in y than ols_yx() below
    N = len(x)
    ((Sxx, Sxy), (Syx, Syy)) = np.cov(x, y, ddof=0) * N
    beta1 = Sxy / Sxx
    x_mean = x.mean()
    y_mean = y.mean()
    alpha1 = ols_alpha(beta1, x_mean, y_mean)
    x_diff = x - x_mean
    y_diff = y - y_mean
    var = np.sum( x_diff**2 * (y_diff - beta1*x_diff)**2)/Sxx**2
    e_beta1 = np.sqrt(var)

    return beta1, alpha1, e_beta1

def ols_yx(x, y):
    # NOTE: this appears to be mislabeled in Isobe 1990 as OLS(Y|X) in Isobe, but is actually OLS(X|Y)
    N = len(x)
    ((Sxx, Sxy), (Syx, Syy)) = np.cov(x, y, ddof=0) * N
    beta2 = Syy / Sxy
    x_mean = x.mean()
    y_mean = y.mean()
    alpha2 = ols_alpha(beta2, x_mean, y_mean)
    x_diff = x - x_mean
    y_diff = y - y_mean
    var = np.sum( y_diff**2 * (y_diff - beta2*x_diff)**2)/Sxy**2
    e_beta2 = np.sqrt(var)
    return beta2, alpha2, e_beta2

def ols_bisector_beta(beta1, beta2):
    beta3 = (beta1*beta2 - 1. + np.sqrt((1. + beta1**2)*(1. + beta2**2)))/(beta1 + beta2)
    return beta3

def ols_bisector_beta_err(x, y, beta1, beta2, e_beta1, e_beta2):
    N = len(x)
    ((Sxx, Sxy), (Syx, Syy)) = np.cov(x, y, ddof=0) * N
    beta3 = ols_bisector_beta(beta1, beta2)
    x_diff = x - np.mean(x)
    y_diff = y - np.mean(y)
    cov =  np.sum( x_diff * y_diff * (y_diff - beta1*x_diff) * (y_diff - beta2*x_diff)) / (beta1*Sxx**2)
    factor = beta3**2 / ((beta1 + beta2)**2 * (1 + beta1**2) * (1 + beta2**2))
    sum = ((1 + beta2**2)**2 * e_beta1**2 +
           (1 + beta1**2)**2 * e_beta2**2 +
           2 * (1 + beta1**2) * (1 + beta2**2) * cov)
    var = factor * sum
    std = np.sqrt(var)
    return std

def ols_bisector_params(beta1, beta2, x_mean, y_mean):
    beta3 = ols_bisector_beta(beta1, beta2)
    alpha3 = ols_alpha(beta3, x_mean, y_mean)
    return [beta3, alpha3]

def ols_bisector(x, y):
    beta1, alpha1, e_beta1 = ols_xy(x, y)
    beta2, alpha2, e_beta2 = ols_yx(x, y)
    (beta3, alpha3) = ols_bisector_params(beta1, beta2, np.mean(x), np.mean(y))
    e_beta3 = ols_bisector_beta_err(x, y, beta1, beta2, e_beta1, e_beta2)
    return beta3, alpha3, e_beta3
    
def ols_bisector_werrs_beta(x, y, e_x, e_y):
    # Feigelson & Babu 1992 equations 1-6, but see erratum
    # TODO: Check if N is needed with cov
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

def _ols_xy_interval(x, y, q, kind):
    """Returns a function giving the +/- confidence interval around the OLS(X|Y) line

    Reference: DeGroot & Schervish "Probability & Statistics" Fourth Edition, 2012, pp 715-716
    """
    n = x.size
    beta1, beta0, std = ols_xy(x, y)
    Y = np.poly1d((beta1, beta0))
    T = scipy.stats.t.ppf(q, df=n-2)
    Ssq = np.power(y - Y(x), 2).sum()
    sigma = np.sqrt(Ssq/(n - 2))
    x_mean = x.mean()
    sx = np.sqrt(np.power((x - x_mean),2).sum())

    if kind == 'confidence':
        return lambda xx: T * sigma * np.sqrt( 1/n + ((xx - x_mean)/sx)**2 )
    elif kind == 'prediction':
        return lambda xx: T * sigma * np.sqrt( 1 + 1/n + ((xx - x_mean)/sx)**2 )
    else:
        raise Exception("invalid interval kind '%s'" % kind)

def ols_xy_confidence(x, y, q):
    return _ols_xy_interval(x, y, q, 'confidence')

def ols_xy_prediction(x, y, q):
    return _ols_xy_interval(x, y, q, 'prediction')
