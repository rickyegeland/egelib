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
