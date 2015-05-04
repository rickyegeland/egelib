import numpy as np

def jiggle(y, yerr):
    """
    Return new array sampled from Gaussians with mean=y[i] and std=yerr[i]
    """
    return np.random.normal(y, yerr)

def remove_outliers(y, pct):
    """
    Remove values from y that are in the top/bottom percentile
    """
    hi_thresh = np.percentile(y, 100 - pct)
    lo_thresh = np.percentile(y, 0 + pct)
    keep = (y >= lo_thresh) & (y <= hi_thresh)
    return y[keep]

def cross_calibrate(y1, y2, c1=None, c2=None):
    """
    Linear cross-calibrate y2 to the scale of y1
    """
    y1c = y1[c1].flatten()
    y2c = y2[c2].flatten()
    # Fitting functions
    fitfunc = lambda p, x: p[0] + p[1] * x
    minfunc = lambda p, x, y: (y - fitfunc(p, x))
    # Calibration function
    calfunc = lambda p, y: (y - p[0])/p[1]
    # Fit once to get the line
    p0 = [0.0, 1.0]
    out = scipy.optimize.leastsq(minfunc, p0, args=(y1c, y2c))
    pfit = out[0]
    return calfunc(pfit, y2), pfit

def cross_calibration_err(y1, y2, e2, c1=None, c2=None, e1=None, Ntrials=5000, Nx=100, xlims=None):
    """
    Use monte carlo to find the uncertainty in a linear cross-calibration of y2 to the scale of y1
    """
    fitfunc = lambda p, x: p[0] + p[1] * x
    if xlims is None:
        xlims = (y1.min(), y1.max())
    linex = np.linspace(xlims[0], xlims[1], Nx)
    ptrials = np.zeros((Ntrials, 2))
    linetrials = np.zeros((Ntrials, linex.size))
    caltrials = np.zeros((Ntrials, y2.size))
    for i in range(Ntrials):
        if e1 is not None:
            y1_jig = jiggle(y1, e1)
        else:
            y1_jig = y1
        y2_jig = jiggle(y2, e2)
        y2_cal, pfit = cross_calibrate(y1_jig, y2_jig, c1, c2)
        linetrials[i] = fitfunc(pfit, linex)
        caltrials[i] = y2_cal
        ptrials[i] = pfit
    calmean = np.mean(caltrials, axis=0)
    calstd = np.std(caltrials, axis=0)
    pmean = np.mean(ptrials, axis=0)
    pstd = np.std(ptrials, axis=0)
    linemean = np.mean(linetrials, axis=0)
    linestd = np.std(linetrials, axis=0)
    return calmean, calstd, pmean, pstd, linex, linemean, linestd
