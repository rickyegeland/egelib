import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import astropy.time

import egelib.timeseries

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

def search_rotations(t, x, e=None, label=None,
                     N_thresh=20, sig_thresh=0.10, sig_mode='interp',
                     sep_thresh=0.20, unc_thresh=0.20,
                     dP=0.01, minP=2., maxP=50.,
                     N_trials=1000, detrend=False):
    ts, xs = egelib.timeseries.seasonal_series(t, x)
    ts, es = egelib.timeseries.seasonal_series(t, e)
    N_seas = len(ts)
    
    periods = np.arange(minP, maxP, dP) / 365.25
    print "dP=%0.3g N_P=%i" % (dP, periods.size)
    results = []
    t_rot = np.zeros(N_seas)
    P_rot = np.ones(N_seas) * np.nan
    e_rot = np.ones(N_seas) * np.nan
    FAP_rot = np.ones(N_seas) * np.nan
    
    if label is None:
        label = "Season %i"
    else:
        label = "%s: Season %%i" % (label)
    for ix, t in enumerate(ts):
        print "===", label % ix, "==="
        t_rot[ix] = t.decimalyear.mean()
        x = xs[ix]
        if detrend:
            x = scipy.signal.detrend(x)
        e = es[ix]
        N = x.size
        
        if N < N_thresh:
            print "skipping: N=%i < thresh=%i" % (N, N_thresh)
            continue

        # Periodogram
        periods, power = egelib.timeseries.lombscargle(t, x, periods=periods)

        # First-pass: use the Horne 1986 estimate
        sig = 1.0 - sig_thresh
        Ni_est = egelib.timeseries.horne1986_Ni(N)
        z_est = egelib.timeseries.FAP_threshold(sig, Ni_est)
        z_thresh = z_est # In first pass, this is the threshold
        zmin = np.linspace(0., 1.2 * z_est, 1000)
        peaks = egelib.timeseries.peak_indices(power, z_est)
        if peaks.size > 0:
            # If a significant peak was found, run a 
            # Monte carlo to better estimate significance threshold
            did_mc = True
            mc = egelib.timeseries.lombscargle_mc(t, N_trials, periods=periods)
            zmin, FAP = egelib.timeseries.FAP_dist(mc['peak_amps'])
            Ni_mc = egelib.timeseries.FAP_fit(zmin, FAP)
            FAPsort = np.argsort(FAP)
            z_interp = np.interp(sig_thresh, FAP[FAPsort], zmin[FAPsort])
            z_fit = egelib.timeseries.FAP_threshold(sig, Ni_mc)
            # Overwrite z_thresh with preferred threshold
            if sig_mode == 'interp':
                z_thresh = z_interp
            elif sig_mode == 'fit':
                z_thresh = z_fit
            elif sig_mode == 'est':
                z_thresh = z_est
            else:
                raise Exception("sig_mode=%s not valid" % sig_mode)
            print "FAP thresh=%0.2g%% Ni_mc=%0.1f Ni_est=%0.1f z_interp=%0.3f z_fit=%0.3f z_est=%0.3f sig_mode=%s" % \
                (sig_thresh*100., Ni_mc, Ni_est, z_interp, z_fit, z_est, sig_mode)
        else:
            did_mc = False
            Ni_mc = None
            FAP = None
            z_interp = None
            z_fit = None

        # Plot 1: Time Series
        plt.figure(figsize=(16, 3))
        plt.subplot(1, 3, 1)
        plt.plot(t.jd - 2400000, x, 'k.')
        plt.gca().ticklabel_format(useOffset=False)
        plt.gca().xaxis.set_major_locator(mpl.ticker.MultipleLocator(50))
        plt.title("%s %.1f-%.1f (N=%i)" % (label % ix, t[0].jyear, t[-1].jyear, N))
        plt.xlabel("BJD")
        plt.ylabel(r"$\Delta$ x")

        # Plot 2: Periodogram
        plt.subplot(1, 3, 2)
        plt.axhline(z_est, color='r', ls='--')
        if did_mc:
            plt.axhline(z_interp, color='gray', ls=':')
            plt.axhline(z_fit, color='r', ls='-')
            plt.axhline(z_thresh, color='b', ls='-')
        plt.plot(periods * 365.25, power, 'k-')
        plt.xlabel("Period [d]")
        plt.ylabel("Power")
        plt.ylim(0, 20) # TODO: configurable

        # Find highest peaks above threshold
        peaks = egelib.timeseries.peak_indices(power, z_thresh)
        peak_found = (peaks.size > 0)
        peaks_all = egelib.timeseries.peak_indices(power)
        peak_period = periods[peaks] * 365.25
        peak_power = power[peaks]
        # Note: Ni_mc from the fit to the mc is often a poor fit near low FAP
        #       Ni_est fares much better there, but is poor everywhere else
        #       It seems like the FAP_model is not actually a good model in many cases...
        #       Using Ni_est to estimate peak FAP.  It may not be reliable at very low values...
        # peak_FAP = egelib.timeseries.FAP_model(peak_power, Ni_mc)
        # peak_FAP = np.interp(peak_power, zmin, FAP) # interpolate?
        peak_FAP = egelib.timeseries.FAP_model(peak_power, Ni_est)

        if peak_found:
            # Print peak period (FAP)
            print "peaks: ",
            for i in range(peaks.size):
                print "%0.3f d (%0.1e)" % (peak_period[i], peak_FAP[i]),
                if ix < (peaks.size - 1): print ", ",
            print
        else:
            print "No significant peaks"

        # Do not count pgrams with a very close secondary peak
        second_peak = False
        if peak_found and peaks_all.size > 1:
            pk1_power = power[peaks_all][0]
            pk2_power = power[peaks_all][1]
            frac = pk2_power/pk1_power
            print "Secondary peak check: p1=%0.3f p2=%0.3f frac=%0.3f thresh=%0.3f" % \
                    (pk1_power, pk2_power, frac, sep_thresh)
            if frac > 1. - sep_thresh:
                second_peak = True
        
        if peak_found and not second_peak:
            best_P = peak_period[0]
            best_FAP = peak_FAP[0]
            plt.title("Periodogram (peak=%0.3f d, FAP=%0.1e)" % (best_P, best_FAP))
            P_rot[ix] = best_P
            FAP_rot[ix] = best_FAP
        elif peak_found and second_peak:
            plt.title("Periodogram (multiple sig. peaks)")
        else:
            plt.title("Periodogram (no sig. peaks)")
        
        # Calculate uncertainty of peak with Monte Carlo
        P_std = None
        if peak_found and not second_peak:
            P = periods[peaks][0]
            sel = (periods <= P+P*unc_thresh) & (periods >= P-P*unc_thresh) # check for periods within unc_thresh %
            mc_periods = periods[sel]
            mc_periods, mc_power = egelib.timeseries.peaks_mc(t, x, e, N_trials=N_trials, N_peaks=1, periods=mc_periods)
            P_d = P * 365.25
            P_mc = mc_periods.mean() * 365.25
            P_std = mc_periods.std() * 365.25
            pct = (P_std / P_mc) * 100.
            e_rot[ix] = P_std
            print "Peak error: P=%0.4g P_mean=%0.4g P_std=%0.4g (%0.4g%%)" % (P_d, P_mc, P_std, pct)
        
        # Plot 3: FAP model fit
        plt.subplot(1, 3, 3)
        plt.plot(zmin, egelib.timeseries.FAP_model(zmin, Ni_est), 'r--')
        if did_mc:
            plt.plot(zmin, FAP, 'k,')
            plt.plot(zmin, egelib.timeseries.FAP_model(zmin, Ni_mc), 'r-')
            plt.plot(z_thresh, sig_thresh, 'b+', ms=10, mew=2)
        plt.title("FAP model fit")
        plt.xlabel("z_min")
        plt.ylabel("P(z_rand > z_min)")
        
        # Return results
        results.append(dict(t=t, x=x, periods=periods, power=power, season=ix,
                            zmin=zmin, FAP=FAP,
                            Ni_mc=Ni_mc, Ni_est=Ni_est, sig=sig, sig_mode=sig_mode,
                            z_interp=z_interp, z_fit=z_fit, z_est=z_est, z_thresh=z_thresh,                            
                            peaks=peaks, peak_period=peak_period, peak_power=peak_power, peak_FAP=peak_FAP,
                            err=P_std))
    
    t_rot = astropy.time.Time(t_rot, format='decimalyear', scale='tcb')
    return t_rot, P_rot, e_rot, FAP_rot, results
