def printstats(x, label, precision=3):
    """Print basic statistics about a 1D array"""
    fmt = "N=%%i min=%%0.%(p)if max=%%0.%(p)if median=%%0.%(p)if mean=%%0.%(p)if std=%%0.%(p)if" % dict(p=precision)
    print label, fmt % \
    (x.size, x.min(), x.max(), np.median(x), x.mean(), x.std())
