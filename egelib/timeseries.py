import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import scipy.signal
import scipy.optimize
import astropy.time
import timeit
import egelib.stats

# TODO: move plotting functions somewhere else

def duration(t, unit=None):
    """
    Duration of an astropy.time.Time object
    
    Inputs:
     - t <astropy.time.Time> : time axis
     - unit <str> : optional unit of duration.  None returns an astropy.time.TimeDelta object.
                    'yr', 'd', 's' return duration as Julian Year, Julian Day, and seconds, respectively.

    Output:
     - Duration of timeseries.  Output type is either astropy.time.TimeDelta or float, depending on 'unit'
    """
    delta = np.max(t) - np.min(t) # TimeDelta object
    if unit is None:
        return delta
    elif unit == 'yr':
        return delta.jd / (365.25)
    elif unit == 'd':
        return delta.jd
    elif unit == 's':
        return delta.sec
    else:
        raise Exception('invalid unit %s' % unit)

def intervals(t, unit=None):
    """
    Returns array of intervals from an astropy.time.Time object
    
    Inputs:
     - t <astropy.time.Time> : time axis
     - unit <str> : optional unit of intervals.  None returns an astropy.time.TimeDelta object.
                    'yr', 'd', 's' return duration as Julian Year, Julian Day, and seconds, respectively.

    Output:
     - Array of intervals from timeseries.
       Output type is either astropy.time.TimeDelta or array of float, depending on 'unit'.
    """
    if unit is None:
        t_sort = astropy.time.Time(np.sort(t))
    elif unit == 'yr':
        t_sort = np.sort(t.jyear)
    elif unit == 'd':
        t_sort = np.sort(t.jd)
    elif unit == 's':
        t_sort = np.sort(t.unix)
    else:
        raise Exception('invalid unit %s' % unit)
    return t_sort[1:] - t_sort[0:-1]

def median_nyquist(t, unit='yr'):
    """Return the "median Nyquist frequency", defined as 0.5/median_interval
    
    'unit' is the inverse unit returned and may be 'yr', 'd', or 's'.

    TODO: document parameters
    """
    ivals = intervals(t, unit)
    f_median_nyquist = 0.5/np.median(ivals) # 1/unit
    return f_median_nyquist

def critical_periods(t, unit='yr'):
    """Compute various critical time periods and frequencies from a timeseries

    TODO: document parameters
    """
    Nsamp = len(t)
    T = duration(t, unit)
    f_nyquist_eq = 0.5/(T/Nsamp)
    f_nyquist_med = median_nyquist(t, unit)
    return { 'duration' : T, 
             'nyquist_eq' : 1.0/f_nyquist_eq,
             'nyquist_med' : 1.0/f_nyquist_med }

def find_coincident(t1, t2, dt, unit='d', unique=True):
    """
    Find coincident samples within dt between two time axes
    """
    if unit == 'yr':
        t1 = t1.jyear
        t2 = t2.jyear
    elif unit == 'd':
        t1 = t1.jd
        t2 = t2.jd
    elif unit == 's':
        t1 = t1.unix
        t2 = t2.unix
    t1_result = []
    t2_result = []
    for i in range(t1.size):
        t = t1[i]
        distances = np.abs(t2 - t)
        coincident = np.where(distances <= dt)[0]
        if unique:
            # take only the closest coincident point
            sort_ixs = np.argsort(distances[coincident])
            coincident = coincident[sort_ixs][0:1]
        for j in coincident:
            t1_result.append(i)
            t2_result.append(j)
    return t1_result, t2_result

def overlap(t1, t2):
    """
    Returns duration (tstart, tend) for which t1 and t2 overlap, or (None, None) if no overlap.
    """
    t1 = dict(min=np.min(t1), max=np.max(t1))
    t2 = dict(min=np.min(t2), max=np.max(t2))
    for t in (t1, t2):
        t['dur'] = t['max'] - t['min']

    # Ensure t1 min < t2 min
    if t2['min'] < t1['min']:
        print 't2 starts earlier'
        t1, t2 = t2, t1
    
    # var names wrt t2
    min_inside = t2['min'] >= t1['min'] and t2['min'] <= t1['max']
    max_inside = t2['max'] <= t1['max']
    if min_inside and max_inside:
        # t2 completely contained by t1
        return (t2['min'], t2['max'])
    elif min_inside:
        # t2 partially contained by t1
        return (t2['min'], t1['max'])
    else:
        # no overlap
        return (None, None)

def overlap_calib(t1, y1, t2, y2):
    """
    Return scaling factor which sets the mean of y2 equal to that of y1
    
    C = y1.mean() / y2.mean() where the mean is for the region in which the 
    two time series overlap.
    """
    tstart, tend = overlap(t1, t2)
    if tstart is None:
        raise Exception('No overlap between time series')
    y1 = y1[(t1 >= tstart) & (t1 <= tend)]
    y2 = y2[(t2 >= tstart) & (t2 <= tend)]
    C = y1.mean() / y2.mean()
    return C

def season_offset(t):
    """
    Return fraction of year which is between observing seasons (bin edge)
    
    Input:
     - t <astropy.time.Time> : time axis
     
    Output:
     - <float> : season offset, in fraction of a Julian year
    
    Observing seasons are portions of the year in which observations 
    are possible.  Between the seasons are gaps with no observations.
    This function returns the point in the year (as a fraction of a year)
    which is the midpoint of the gaps
    """
    t_sort = np.sort(t) # sorted copy
    delta = t_sort[-1] - t_sort[0]
    seconds_in_year = 365.25 * 86400
    Nyears = int(delta.sec / seconds_in_year)
    f = np.vectorize(lambda x: x.sec) # function to turn TimeDelta into float seconds
    dt = f(t_sort[1:] - t_sort[0:-1])   # ... use the above
    gaps = np.sort(dt)[-Nyears:] # use sorted copy
    median_gap = np.median(gaps)
    offset = median_gap / 2 # half-width of gap in seconds
    # Find index of gap closest to mean gap
    min_diff = np.inf
    i_median_gap = -1
    for i in range(dt.size):
        diff = np.abs(dt[i] - median_gap)
        if diff < min_diff:
            min_diff = diff
            i_median_gap = i
    before_gap = t_sort[i_median_gap]
    offset_frac = (before_gap.jyear + offset/seconds_in_year) % 1
    return offset_frac

def season_edges(t):
    """
    Given a time axis, return the edges between observing seasons.
    """
    offset = season_offset(t)
    yr_min = t.datetime.min().year
    left_frac = t.jyear.min() % yr_min
    if left_frac < offset:
        ex_left = 1
    else:
        ex_left = 0
    edges = np.arange(yr_min - ex_left + offset, t.jyear.max() + 1, 1.0)
    return astropy.time.Time(edges, format='jyear')

def plot_edges(t, y, label):
    """
    Plot data and season edges together, to verify binning.
    """
    edges = season_edges(t)
    plt.figure()
    plt.plot(t.datetime, y, '.')
    for e in edges.datetime:
        plt.axvline(e)
    plt.title('%s Season Edges' % label)
    plt.margins(0.05)

def season_indices(t, edges=None, hard=False):
    """
    Return indice array for which maps time sample to a season

    Example:
    >>> seasons = season_indexes(t)
    >>> for i in range(len(t)):
    >>>     print "Time point %i at %s is in season %i" % (i, t[i], seasons[i])
    """
    if edges is None:
        edges = season_edges(t)
    season_ixs = np.digitize(t.jyear, edges.jyear)
    N = season_ixs.size
    seasons = []
    for ix in np.unique(season_ixs):
        season_members = np.arange(N)[season_ixs == ix] # bool array to index array
        seasons.append(season_members)
    if hard:
        if t[seasons[0]].jyear.max() < edges[0].jyear:
            seasons = seasons[1:]
        if t[seasons[-1]].jyear.min() > edges[-1].jyear:
            seasons = seasons[:-1]
    return seasons

def seasonal_series(t, y, edges=None, hard=False):
    """
    Return array of time series for each season.
    
    Example:
    >>> ts, ys = seasonal_series(t, y)
    >>> for i in range(len(ts)):
    >>>     print "Season %i has time samples %s and data %s" % (i, ts[i], ys[i])
    """
    season_ixs = season_indices(t, edges=edges, hard=hard)
    ts = []
    ys = []
    for season in season_ixs:
        ts.append(astropy.time.Time(t.jyear[season], format='jyear', scale=t.scale))
        ys.append(y[season])
    return ts, ys

def seasonal_means(t, y, edges=None, hard=False):
    """
    Calculate seasonal means for some timeseries data
    
    Input:
     - t <astropy.time.Time> : time axis
     - y <numpy.ndarray>     : data axis
     
    Output:
     - <astropy.time.Time>   : seasonal mean time
     - <numpy.ndarray>       : mean of y for each season
     - <numpy.ndarray>       : standard deviation of y in each season
     - <numpy.ndarray>       : count of measurements in each season
    """
    ts, ys = seasonal_series(t, y, edges=edges, hard=hard)
    t_means = [t.jyear.mean() for t in ts]
    t_means = astropy.time.Time(t_means, format='jyear', scale=t.scale)
    y_means = np.array([y.mean() for y in ys])
    y_std = np.array([y.std() for y in ys])
    y_N = np.array([y.size for y in ys])
    return t_means, y_means, y_std, y_N

def seasonal_calc(t, y, func, edges=None):
    """
    Calculate seasonal means for some timeseries data
    
    Input:
     - t <astropy.time.Time> : time axis
     - y <numpy.ndarray>     : data axis
     - func <function>       : function which takes array of y
     
    Output:
     - <astropy.time.Time>   : seasonal mean time
     - <numpy.ndarray>       : func(y) for each season
    """
    ts, ys = seasonal_series(t, y, edges=edges)
    t_means = [t.jyear.mean() for t in ts]
    t_means = astropy.time.Time(t_means, format='jyear', scale=t.scale)
    f_y = np.array([func(y) for y in ys])
    return t_means, f_y

def seasonal_mad_outliers(t, x, edges=None, seasons=None, bool=False, global_mad=True, Nseas=10, Nsigma=4):
    out_ixs = []
    if seasons is None and edges is None:
        edges = season_edges(t)
        seasons = season_indices(t, edges=edges)
    elif seasons is None and edges is not None:
        seasons = season_indices(t, edges=edges)

    if global_mad:
        # Global seasonal threshold
        mads = []
        for iseas, s in enumerate(seasons):
            if len(s) == 0 or len(s) < Nseas: continue
            xs = x[s]
            mads.append(egelib.stats.mad_sigma(xs))
        medmad = np.median(mads)
        for iseas, s in enumerate(seasons):
            if len(s) == 0: continue
            xs = x[s]
            med = np.median(xs)
            out = np.abs(xs - med) > Nsigma * medmad
            sout = s[out]
            out_ixs.extend(sout)
    else:
        # Local seasonal threshold
        for iseas, s in enumerate(seasons):
            if len(s) == 0 or len(s) < Nseas: continue
            xs = x[s]
            out = egelib.stats.mad_outliers(xs, Nsigma=Nsigma)
            sout = s[out]
            out_ixs.extend(sout)
    out_ixs = np.unique(np.array(out_ixs))
    if bool is True:
        out = np.zeros(len(t), dtype='bool')
        if out_ixs.size > 0:
            out[out_ixs] = True
        return out
    else:
        return out_ixs

def mad_outliers(t, x, w, bool=False, Nsigma=4):
    """Identify outliers using the median absolute deviation"""
    x_med = local_func(np.median, t, x, w)
    x_mad = local_func(egelib.stats.mad_sigma, t, x, w)
    out = np.abs(x - x_med) >= Nsigma * x_mad
    if bool is True:
        return out
    else:
        return np.arange(len(out))[out] # bool to index

def sigmanorm(y):
    """Rescale data to units of its standard deviation"""
    y = y.copy()
    y -= y.mean() # set to zero mean
    y /= y.std()  # rescale to units of sigma
    return y

def lombscargle(t, y, periods=None, unit='yr', density_method='nyquist_med',
                minP=None, maxP=None, Nout=None, ofac=1.0, hifac=1.0, amp_norm=False, apply_sigmanorm=True):
    """
    Compute the Lomb-Scargle periodogram
    
    Inputs:
     - t <astropy.time.Time> : time axis
     - y <numpy.ndarray> : data axis
     - periods <numpy.ndarray> : (optional) periods of the output periodogram.  If None the periods will be 
                                  determined according to density_method
     - unit <str> : unit of t, and unit of the period axis on output
     - density_method <str> : Method used to determine the output spectral density (periods)
     - minP <float> : minimum period to output
     - maxP <float> : maximum period to output
     - Nout <int> : number of periods to output
     - ofac <float> : oversampling factor
     - hifac <float> : high frequency factor, defined as f_hi/f_nyquist
     - amp_norm <bool> : whether to normalize the amplitudes
     - apply_sigmanorm <bool> : rescale data to units of its standard deviation
     
    Outputs:
     - <numpy.ndarray> : periods
     - <numpy.ndarray> : spectral power of Lomb-Scargle periodogram, normalized by the variance of y
     
    References:
     - Horne & Baliunas 1986
     - Press et al. 1988 "Numerical Recipes in C: The Art of Scientific Computing" Second edition.
     - scipy.signal.lombscargle documentation
    
    Valid density methods:
     - 'nyquist_med' : Periods are intervals of 1/median_nyquist_frequency.  See median_nyquist()
     - 'oversample'  : Periods are intervals of 1/scaled_nyquist_frequency.  Same method as used in 
                       Numerical Recipes.
     - 'equalP'      : Simple equidistand periods from minP to maxP with Nout divisions.
     
    Note that the default density method, 'nyquist_med', is computationally expensive.
    When doing many periodograms, set the 'periods' keyword for best performance.
    """
    if periods is None:
        Nsamp = t.value.size
        T = duration(t, unit)
        f_Nyquist = 0.5 * Nsamp/T

        if density_method == 'nyquist_med':
            # Intervals of the "median nyquist period"
            f_nyquist_med = median_nyquist(t, unit)
            deltaP = 1.0 / (f_nyquist_med * ofac)
            if minP is None:
                minP = deltaP # TODO: determine in terms of hifac instead
            if maxP is None:
                maxP = T
            periods = np.arange(minP, maxP, deltaP)
            freqs = 2*np.pi/periods
        elif density_method == 'oversample':
            # Method described in numerical recipes
            # Equal-spaced frequencies based on Nyquist freq with oversampling factor
            delta_freq = 1.0/(T*ofac)
            if maxP is None:
                lofreq = 1.0/T
            else:
                lofreq = 1.0/maxP
            f_Nyquist = 0.5 * Nsamp/T
            if minP is None:
                hifreq = hifac * f_Nyquist
            else:
                hifreq = 1/minP
            freqs = np.arange(lofreq, hifreq, delta_freq)
            periods = 1/freqs
            freqs *= 2*np.pi # to angular freqs
        elif density_method == 'equalP':
            Pcrit = critical_periods(t, unit)
            if minP is None:
                minP = 1.0/f_Nyquist
            if maxP is None:
                maxP = T
            if Nout is None:
                Nout = Nsamp
            periods = np.linspace(minP, maxP, Nsamp)
            freqs = 2*np.pi/periods
        else:
            raise Exception("invalid dansity_method '%s'" % density_method)
    else:
        freqs = 2*np.pi/periods

    if apply_sigmanorm:
        y = sigmanorm(y)
    
    # Calculate periodogram
    # must send ndarrays, not datetime object or Column object
    if unit == 'yr':
        t = t.jyear
    elif unit == 'd':
        t = t.jd
    pgram = scipy.signal.lombscargle(t, y.astype('float64'), freqs)
    if amp_norm:
        normval = Nsamp/4.0 # value to normalize scipy.signal.lombscargle() result
        pgram = np.sqrt(pgram/normval)
    return periods, pgram

def peak_indices(pgram, thresh=0):
    """Find peaks above thresh, and return peak indexes in order of decreasing power"""
    diff = pgram[1:] - pgram[0:-1]
    peaks_offset1 = (diff[0:-1] > 0) & (diff[1:] < 0)
    peaks = np.insert(peaks_offset1, 0, False) # fix the offset; the first point cannot be a peak
    peaks = np.insert(peaks, -1, False)        # ... nor the last point
    peaks = peaks & (pgram >= thresh) # apply the threshold
    peak_ixs = np.arange(peaks.size)[peaks] # convert bool array to index list
    order = np.argsort(pgram[peak_ixs])[::-1] # find order of decreasing power
    ordered_peaks = peak_ixs[order] # reorder the peak indices to decreasing power
    return ordered_peaks

def peaks(periods, pgram, thresh=0):
    peaks = peak_indices(pgram, thresh=thresh)
    return periods[peaks]

def peak_uncertainty(t, S, P):
    """Uncertainty estimate for peak period P found in time series.

    Ref:
      Kovacs 1981
      Baliunas et al. 1985
      Horne & Baliunas 1986
    """
    t = t.jyear
    N = S.size
    T = t.max() - t.min()
    amp, phase, offset = find_sine_params(t, S - S.mean(), P)
    sine = sinefunc(t, P, amp, phase, offset)
    residual = (S - S.mean()) - sine
    sigma = residual.std()
    #print "P=%0.3f N=%i T=%0.1f amp=%0.3g sigma=%0.3g" % (P, N, T, amp, sigma)
    return 3. * sigma * P**2 / (4. * T * amp * np.sqrt(N))

#
# Monte Carlo
#
def shuffled_lombscargle(t, y, **kwargs):
    y_shuffled = np.random.permutation(y)
    periods, pgram_shuffled = lombscargle(t, y_shuffled, **kwargs)
    return y_shuffled, periods, pgram_shuffled

def random_lombscargle(t, **kwargs):
    y_random = np.random.normal(size=t.value.shape[0])
    periods, pgram_random = lombscargle(t, y_random, **kwargs)
    return y_random, periods, pgram_random

def lombscargle_mc(t, Ntrials, y=None, **kwargs):
    if y is None:
        mode = "random"
        do_trial = lambda **kwargs: random_lombscargle(t, **kwargs)
    else:
        mode = "shuffle"
        do_trial = lambda **kwargs: shuffled_lombscargle(t, y, **kwargs)

    # A test run, just to figure out how many periods we'll get
    trial_y, periods, trial_pgram = do_trial(**kwargs)

    peak_periods = np.zeros(Ntrials)
    peak_amps = np.zeros(Ntrials)
    sum_y = np.zeros_like(t.jyear)
    sum_pgram = np.zeros_like(periods)
    print "Starting %s monte carlo, %i trials of %i points to %i periods..." % \
    (mode, Ntrials, t.jyear.size, periods.size)
    print "Progress:",
    for i in range(Ntrials):
        if i > 0 and (i+1) % 1000 == 0:
            print 'k',
        trial_y, periods, trial_pgram = do_trial(periods=periods)
        imax = np.argmax(trial_pgram)
        peak_periods[i]= periods[imax]
        peak_amps[i] = trial_pgram[imax]
        sum_y += trial_y
        sum_pgram += trial_pgram
    print "...Done"
    mean_y = sum_y / Ntrials
    mean_pgram = sum_pgram / Ntrials
    results = { 'mode' : mode,
                'Ntrials' : Ntrials,
                't' : t,
                'mean_y' : mean_y,
                'periods' : periods,
                'mean_pgram' : mean_pgram,
                'peak_periods' : peak_periods,
                'peak_amps' : peak_amps,
                'kwargs' : kwargs }
    return results

def peaks_mc(t, y, e, thresh=0, N_trials=5000, N_peaks=None, **pgram_kwargs):
    """Obtain distribution of peak periods and L-S power given data with uncertainties"""
    tstart = timeit.time.clock()

    def do_trial(**kwargs):
        y_jig = np.random.normal(y, e)
        periods, power = lombscargle(t, y_jig, **kwargs)
        peaks = peak_indices(power, thresh=thresh)
        pk_periods = periods[peaks]
        pk_power = power[peaks]
        if N_peaks is not None and pk_periods.size >= N_peaks:
            pk_periods = pk_periods[0:N_peaks]
            pk_power = power[0:N_peaks]
        return periods, pk_periods, pk_power
        
    # Do one trial to get the periods
    periods, mc_pk_periods, mc_pk_power = do_trial(**pgram_kwargs)
    
    # Now do the rest
    for i in range(N_trials - 1):
        periods, pk_periods, pk_power = do_trial(periods=periods)
        mc_pk_periods = np.append(mc_pk_periods, pk_periods)
        mc_pk_power = np.append(mc_pk_power, pk_power)
    tend = timeit.time.clock()
    print "trials=%i peaks=%i thresh=%0.3g" % (N_trials, mc_pk_periods.size, thresh)
    print "%i trials of %i samples to %i periods in %f s" % \
        (N_trials, y.size, periods.size, tend - tstart)
    return mc_pk_periods, mc_pk_power

def plot_mc(name, mc_results):
    mode = mc_results['mode'].capitalize()
    plot_pgram(mc_results['t'].jyear, mc_results['mean_y'], mc_results['periods'], mc_results['mean_pgram'],
               '%s Mean %s Time Series (N=%i)' % (name, mode, Ntrials))

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.hist(1.0/mc_results['peak_periods'], 20, normed=True)
    plt.title('%s Peak Frequency of %s Trials' % (name, mode))
    plt.subplot(1, 2, 2)
    plt.hist(mc_results['peak_amps'], 20, normed=True)
    plt.title('%s Amplitude of Peak Period of %s Trials' % (name, mode))

    plt.figure()
    plt.plot(mc_results['peak_periods'], mc_results['peak_amps'], '.')
    plt.title('%s Peak Amplitudes of N=%i %s Trials' % (name, Ntrials, MODE))

    statdump(mc_results['peak_periods'], '%s periods [yr]' % name)
    statdump(mc_results['peak_amps'], '%s amps [power]' % name)
    print "%s median of mean %s pgram: %0.3f" % (name, mode, np.median(mc_results['mean_pgram']))

#
# Lomb-Scargle False Alarm Probability
#

def FAP_dist(peak_amps, Nbins=1000):
    """False Alarm Probability distribution as a function of power z"""
    zmin = np.linspace(0, peak_amps.max(), Nbins)
    FAP = np.zeros(Nbins)
    for i in range(Nbins):
        FAP[i] = (peak_amps > zmin[i]).sum()
    FAP /= peak_amps.shape[0] # Normalize
    return zmin, FAP

def FAP_model(z, Ni):
    """False Alarm Probability model function"""
    return 1 - (1-np.exp(-z))**Ni

def FAP_fit(zmin, FAP):
    """Fit the FAP model to monte carlo data""" 
    popt, pcov = scipy.optimize.curve_fit(FAP_model, zmin, FAP, p0=[10])
    Ni = popt[0]
    return Ni

def FAP_threshold(sig, Ni):
    """Threshold power above which signals are significant"""
    # F := false alarm probability
    # sig := 1 - F
    return - np.log(1 - sig**(1/Ni))

def FAP_sig(z, Ni):
    """Given a pgram power, return the significance level"""
    return 1 - FAP_model(z, Ni)

def horne1986_Ni(N):
    """Given the number of samples, empirically estimate Ni independent frequencies"""
    return -6.362 + 1.193*N + 0.00098*N**2

def horne1986_FAP_threshold(sig, N):
    """Given the number of samples, empirically estimate significance threhsold"""
    Ni = horne1986_Ni(N)
    return FAP_threshold(sig, Ni)

#
# Sine models
#
def sinefunc(t, P, amp=1.0, phase=0.0, offset=0.0):
    """Model sine wave
    
    Units of t, P, and 
    """
    return np.abs(amp) * np.sin(2*np.pi*t/P + 2*np.pi*phase/P ) + offset

def find_sine_params(t, y, P):
    """Find amplitude, phase, and offset of sine of period P in time series"""
    y = y - y.mean()
    amp0 = np.max([np.abs(y.max()), np.abs(y.min())])
    phase0 = 0.0
    offset0 = 0.0
    def f(t, amp, phase, offset):
        return sinefunc(t, P, amp, phase, offset)
    popt, pcov = scipy.optimize.curve_fit(f, t, y, (amp0, phase0, offset0))
    return popt

def find_sine(t, y, P, t_out=None):
    """Return fittet sine function of period P from the time series"""
    if t_out is None:
        t_out = t
    amp, phase, offset = find_sine_params(t, y, P)
    return sinefunc(t_out, P, amp, phase, offset)

def subtract_sine(t, y, P):
    """Remove a sine of period P from the data"""
    f = np.zeros(y.size)
    amp, phase, offset = find_sine_params(t.jyear, y, P)
    params = (P, amp, phase, offset)
    f += sinefunc(t.jyear, P, amp, phase, offset)
    return y - f, params

def sum_sines(t, param_set):
    """Evaluate the sum of a set of sines on a time axis"""
    f = np.zeros(t.jyear.size)
    for params in param_set:
        f += sinefunc(t.jyear, *params)
    return f

def residual_lombscargle(t, y, P, **kwargs):
    """Lomb-Scargle periodogram with period P removed from time series"""
    sine = find_sine(t.jyear, y, P)
    ynew = y - sine
    return lombscargle(t, ynew, **kwargs)

def samesamp_lombscargle(t, y, P, **kwargs):
    """Lomb-Scargle periodogram of sine of period P with same sampling as time series"""
    sine = find_sine(t.jyear, y, P)
    return lombscargle(t, sine, **kwargs)

def lowpass_model(t, y, lowthresh=10.0, sigthresh=10.0, max_iter=3, Pres=1000., plot_steps=True):
    """Find a model of the low frequency behavior of the given data
    
    The model parameters returned may be used with sum_sines() to
    evaluate the model.
    """
    def plot_pgram(t, y, periods=None, title=None):
        periods, power = pgram(t, y, periods=periods)
        plt.figure(figsize=(16,3))
        ax = plt.gca()
        ax.plot(periods, power, 'k-')
        if title is not None:
            ax.set_title(title)
        ax.set_xlabel('Period [yr]')
        ax.set_ylabel('Power')
    
    if plot_steps:
        plot_pgram(t, y, title='Original (std=%0.3g)' % y.std())

    y_hipass = y
    model_params = []
    newP = np.arange(lowthresh, duration(t, 'yr'), lowthresh/Pres)
    for i in range(max_iter):
        newP, power = pgram(t, y_hipass, periods=newP)
        pk_ixs = peak_indices(power, thresh=sigthresh)
        if pk_ixs.size == 0:
            break
        peaks = np.sort(newP[pk_ixs])[::-1]
        pk = peaks[0]
        y_hipass, subparams = subtract_sine(t, y_hipass, pk)
        model_params.append(subparams)
        if plot_steps:
            plot_pgram(t, y_hipass, title='Round %i result: subtracted %0.3f (std=%0.3g)' % (i, pk, y_hipass.std()))

    if plot_steps:
        model_points = y.size
        t_model = np.linspace(t.jyear.min(), t.jyear.max(), model_points)
        t_model = astropy.time.Time(t_model, format='jyear', scale='utc')
        y_model = sum_sines(t_model, model_params)
        y_model += y.mean()
        y_residual = y - sum_sines(t, model_params)
        plot_pgram(t_model, y_model, title='Lo-Pass Model')
        plot_pgram(t, y_residual, title='Original - Lo-Pass Model (std=%0.3g)' % (y_residual.std()))

    return model_params

#
# Short-Time Lomb Scargle Analysis
#
def STLS():
    # TODO: need to generalize and simplify this
    def plot_pgram(t, S, periods=None, title=None):
        # Whole-series periodogram
        periods, power = pgram(t, S, periods=periods)
        plt.figure(figsize=(16,4))
        ax = plt.gca()
        ax.plot(periods, power, 'k-')
        if title is not None:
            ax.set_title(title)
        ax.set_xlabel('Period [yr]')
        ax.set_ylabel('Power')
        
    t = t_all
    S = S_all
    edges = season_edges(t).jyear
    Nseasons = 5
    dP = 0.005 # years
    minP = 1.1 # years
    maxP = Nseasons/2. # years
    periods = np.arange(minP, maxP, dP)
    print "Period search: dP=%0.3f minP=%0.3f maxP=%0.3f N_P=%i" % (dP, minP, maxP, periods.size)
    
    plot_pgram(t, S, periods, title='Original')
    S_lowpass = sum_sines(t, lowpass_model_params) + S_all.mean()
    S = S - S_lowpass   
    plot_pgram(t, S, periods, title='Residual')
        
    Nwindows = len(edges) - Nseasons
    Npoints = np.zeros((Nwindows))
    var = np.zeros((Nwindows))
    tcenter = np.zeros((Nwindows))
    z_thresh = np.zeros((Nwindows))
    pk_params = np.zeros((Nwindows, 5))
    Pmatrix = np.zeros((Nwindows, periods.size))
    
    # Equal-interval least-squares params search
    lsq_periods = np.arange(minP, maxP, 0.1)
    lsq_matrix = np.zeros((Nwindows, lsq_periods.size, 4))
    print "Equi-interval Phase search: n_periods=%i n_elem=%i" % (lsq_periods.size, lsq_matrix.size)
    
    # Significant peaks least-squares params search
    sigranks, sigpeaks = get_peak_range(minP, maxP)
    sig_lsq_matrix = np.zeros((Nwindows, sigpeaks.size, 4))
    print "Sig-Peak Phase search: n_periods=%i n_elem=%i" % (sigpeaks.size, sig_lsq_matrix.size)

    for ix in range(0, len(edges) - Nseasons):
        # Time Series
        tstart = edges[ix]
        tstop = edges[ix + Nseasons]
        tcenter[ix] = tstart + (tstop - tstart) / 2.
        sel = (t_all.jyear >= tstart) & (t_all.jyear <= tstop)
        t_win = t[sel]
        S_win = S[sel]
        N = S_win.size
        Npoints[ix] = N
        var[ix] = np.var(S_win)
        label = "%02i %0.1f N=%i var=%0.3g" % (ix, tcenter[ix], Npoints[ix], var[ix])
        print label
        
        # Time Series
        plt.figure(figsize=(9,2))
        ax1 = plt.subplot(1,2,1)
        ax1.plot(t_win.jyear, S_win, 'k.')
        ax1.set_title(label)
        ax1.set_xlabel("Year")
        ax1.set_ylabel("S")
        ax1.xaxis.get_major_formatter().set_useOffset(False)
        ax1.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        
        # Periodogram
        ax2 = plt.subplot(1,2,2)
        periods, power = pgram(t_win, S_win, periods=periods)
        Pmatrix[ix] = power
        peaks = peak_indices(power)
        if peaks.size > 0:
            pk_period = periods[peaks][0]
            pk_power = power[peaks][0]
            title = "peak=%0.3f" % pk_period
        else:
            pk_period = None
            title = "no peak"
        ax2.plot(periods, power, 'k-')
        ax2.plot(periods[peaks], power[peaks], 'bd', ms=3)
        ax2.set_title(title)
        ax2.set_xlabel('Period [yr]')
        ax2.set_ylabel('Power')
        
        # Significance
        mc = pgram_mc(t_win, 2000, periods=periods)
        zmin, FAP = FAP_mc(mc['peak_amps'])
        Ni = FAP_fit(zmin, FAP)
        thresh = 0.001
        sig = 1.0 - thresh
        z_thresh[ix] = FAP_threshold(sig, Ni)
        print "FAP Ni=%0.3f thresh=%0.2g%% z_thresh=%0.3f z_est=%0.3f" % (Ni, thresh*100., z_thresh[ix], FAP_threshold(sig, N/2.))
        plt.axhline(z_thresh[ix], color='r')
        
        # Peak phase, Amplitude
        if pk_period is not None:
            amp, phase, offset = find_sine_params(t_win.jyear, S_win, pk_period)
            t_sin = np.linspace(t_win.jyear.min(), t_win.jyear.max(), 100)
            y_sin = sinefunc(t_sin, pk_period, amp, phase, offset)
            ax1.plot(t_sin, y_sin + S_win.mean(), 'b-')
            pk_params[ix] = [pk_period, pk_power, amp, phase, offset]
            print "peak amp phase offset:", pk_params[ix]
            
        # Equi-Interval Least squares phase search
        for j, P in enumerate(lsq_periods):
            amp, phase, offset = find_sine_params(t_win.jyear, S_win, P)
            lsq_matrix[ix,j] = [P, amp, phase, offset]
            
        # Sig-Peaks Least Squares phase search
        for j, P in enumerate(sigpeaks):
            try:
                amp, phase, offset = find_sine_params(t_win.jyear, S_win, P)
            except RuntimeError as e:
                print "ERROR: least-squares fit for P=%0.3f failed:" % P, e
                sig_lsq_matrix[ix,j] = np.ones(4) * np.nan
                continue
            sig_lsq_matrix[ix,j] = [P, amp, phase, offset]
        
    return (tcenter, periods, Pmatrix, Npoints, var, z_thresh, pk_params, lsq_matrix, sig_lsq_matrix)

def plot_STLS(stls_result):
    # TODO: need to generalize and simplify this
    (tcenter, periods, Pmatrix, Npoints, var, z_thresh, pk_params, lsq_matrix, sig_lsq_matrix) = stls_result
    t_peaks1 = []
    t_peaks2 = []
    peaks1 = []
    peaks2 = []
    for ix in range(Pmatrix.shape[0]):
        pk_ixs = peak_indices(Pmatrix[ix], z_thresh[ix])
        pks = periods[pk_ixs]
        t = tcenter[ix]
        #print "ix=%i t=%0.1f pks=%s" % (ix, t, pks)
        if pks.size > 0:
            t_peaks1.append(t)
            peaks1.append(pks[0])
        if pks.size > 1:
            t_peaks2.append([t])
            peaks2.append(pks[1])
        
    X, Y = np.meshgrid(tcenter, periods)
    fig = plt.figure(figsize=(16,8))
    gs = matplotlib.gridspec.GridSpec(3, 3, height_ratios=[1,5,1], width_ratios=[20, 1, 4])
    
    labelfont = 20
    tickfont = 16
    ticklength = 5
    tickwidth = 1
    Ncontlevels = 100
    contmax = 50
    #timelim = (X.min(), X.max())
    timelim = (1969.0, 2015.0)
    def tick_params(ax):
        ax.tick_params(axis='both', which='major', labelsize=tickfont, length=ticklength+4, width=tickwidth)
        ax.tick_params(axis='both', which='minor', length=ticklength, width=tickwidth)
    
    # Contour Plot
    ax1 = plt.subplot(gs[3])
    
    Z = Pmatrix.transpose() / Npoints
    print "Z min=%0.3g max=%0.3g" % (Z.min(), Z.max())
    cont = ax1.contourf(X, Y, Z, levels=np.linspace(0,Z.max(),Ncontlevels), cmap='gray')
    #ax1.contour(X, Y, Z, [z_thresh.mean()], colors=['c'], linestyles=[':'])
    
    for tmin in (1977.1, 1986.7, 1998.4, 2013.9):        
        ax1.axvline(tmin, color='b', linewidth=2, linestyle=':')
    for tmax in (1969.0, 1981.2, 1993.7, 2005.8):
        ax1.axvline(tmax, color='r', linewidth=2, linestyle='--')
    
    Z2 = Pmatrix.transpose()/z_thresh  # units of 99.9% sig thresh
    #cont = ax1.contourf(X, Y, Z, levels=np.linspace(0,Z.max(),Ncontlevels), cmap='gray')
    ax1.contour(X, Y, Z2, [1.], colors=['w'], linestyles=[':'])
    insert(cont) # avoid bad PDF rendering
    ax1.set_xlim(timelim)
    ax1.set_ylim(Y.min(), Y.max())
    ax1.plot(t_peaks1, peaks1, 'wo', ms=8)
    ax1.plot(t_peaks2, peaks2, 'wo', ms=5)
    ax1.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    ax1.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
    ax1.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
    ax1.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
    tick_params(ax1)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.tick_params(axis='x', which='both', color='w')
    ax1.set_ylabel("Period [yr]", fontsize=labelfont)
    plt.subplots_adjust(hspace=0.1, wspace=0.02)
      
    # Color Bar
    axc = plt.subplot(gs[4])
    cbar = fig.colorbar(cont, cax=axc, use_gridspec=True, ticks=np.arange(0, 0.23, 0.02))
    cbar.solids.set_rasterized(True) # avoid bad PDF rendering
    cbar.ax.tick_params(which='major', labelsize=tickfont, length=ticklength+4, width=tickwidth) 
    cbar.set_label("Normalized Power", fontsize=labelfont, rotation=270, labelpad=25)
    
    # Integration Plot
    Zint = Z.sum(axis=1)/Z.shape[1]
    print "Zint min=%0.3g max=%0.3g" % (Zint.min(), Zint.max())
    ax2 = plt.subplot(gs[5], sharey=ax1)
    pos1 = ax2.get_position() # get the original position 
    pos2 = [pos1.x0 + 0.055, pos1.y0,  pos1.width/1.2, pos1.height] 
    ax2.set_position(pos2) # set a new position
    ax2.set_ylim(ax1.get_ylim())
    ax2.plot(Zint, periods, 'k-')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel("Period [yr]", rotation=270, labelpad=25, fontsize=labelfont)
    tick_params(ax2)
    ax2.set_xlim((0.03,0.08))
    ax2.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.04))
    ax2.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.01))
    ax2.set_xlabel("Integrated Power", fontsize=labelfont)

    # Npoints Plot
    ax3 = plt.subplot(gs[0], sharex=ax1)
    ax3.set_xlim(timelim)
    ax3.plot(tcenter, Npoints, 'k.-')
    ax3.set_ylabel('N', fontsize=labelfont)
    ax3.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(150))
    ax3.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(50))
    tick_params(ax3)
    plt.setp(ax3.get_xticklabels(), visible=False)
    
    # Amplitude Plot
    ax4 = plt.subplot(gs[6], sharex=ax3)
    ax4.set_xlim(timelim)
    periods = pk_params[:,0]
    power = pk_params[:,1]
    amps = 2.* np.abs(pk_params[:,2]) # convert to peak-to-peak
    sel = (power > z_thresh) & (amps > 0.001)
    #sel = (power > z_thresh) & (amps > 0.001) & (periods <= 2.0)
    #sel = (power > z_thresh) & (amps > 0.001) & (tcenter > 1977) & (tcenter < 1986.5)
    Acyc = 2.*amps[sel] / S_all.mean()
    statdump(periods[sel], "periods   ", precision=3)
    statdump(amps[sel],    "amplitudes", precision=5)
    statdump(Acyc,         "Acyc      ", precision=5)
    t = t_all
    S = S_all
    S_lowpass = sum_sines(t, lowpass_model_params) + S_all.mean()
    S_hipass = S - S_lowpass
    t_model = astropy.time.Time(np.linspace(timelim[0], timelim[1], 1000), format='jyear', scale='utc')
    S_lowpass = sum_sines(t_model, lowpass_model_params) + S_all.mean()
    ax4.plot(t_model.jyear, S_lowpass - S_model_min, 'r-')
    ax4.plot(tcenter[sel], amps[sel], 'wo', ms=8)
    ax4.set_xlabel("5-yr Window Center", fontsize=labelfont)
    ax4.set_ylabel(r'$\Delta$ S', fontsize=labelfont)
    ax4.set_ylim(0.,0.05)
    ax4.set_yticks([0.01, 0.03, 0.05])
    tick_params(ax4)

    #fig.tight_layout()
    fig.savefig('stls.pdf', bbox_inches='tight', dpi=300)



#
# FFT functions
# TODO: make work with astropy.time axis; unify call signature and output with lombscargle
#       that is, implement similar oversampling techniques and return period axis
#
def freqvals(t):
    """Return (N, period, interval, nyquist freq, low frequency) from time axis"""
    N = len(t)
    T = t[-1] - t[0]
    dt = T/N
    nyquist = 1/(2.0*dt)
    lowfreq = 1/T
    return (N, T, dt, nyquist, lowfreq)

def oversample_NFreq(t, oversample):
    """Compute number of frequencies to use when oversampling a FFT"""
    (N, T, dt, nyquist, lowfreq) = freqvals(t)    
    return 2 * int((nyquist - lowfreq) / ( lowfreq / oversample) + 1) + 1

def fft(t, signal, oversample=1, minP=None, maxP=None):
    # Determine frequencies
    (N, T, dt, nyquist, lowfreq) = freqvals(t)

    # Do "smoothing" of the FFT by zero-padding the signal.  This
    # results in more frequencies in the output.  The number of
    # frequencies we will obtain is determined by the range of valid
    # frequencies from lowfreq to nyquist, divided by steps of the
    # lowfreq shortened by an "oversampling" factor. The factor of 
    # two is due to the fact that the signal is real and negative
    # frequencies will be removed later.
    # When oversample=1 => Nfreq = N and there is no smoothing.
    Nfreq = oversample_NFreq(t, oversample)

    # Calculate fft power.
    # Normalization by N makes power equivilant with periodogram output.
    power = np.abs( scipy.fftpack.fft(signal, n=Nfreq) )**2 / N
    freqs = scipy.fftpack.fftfreq(Nfreq, dt)
    
    # Chop off negative frequencies
    pos = freqs > 0
    freqs = freqs[pos]
    power = power[pos]
    periods = 1.0/freqs
    
    if maxP is not None:
        lopass = periods >= maxP
        freqs = freqs[lopass]
        periods = periods[lopass]
        power = power[lopass]
    
    if minP is None:
        # Chop off frequencies below 1 / T
        hipass = freqs >= lowfreq
    else:
        hipass = periods >= minP
    freqs = freqs[hipass]
    periods = periods[hipass]
    power = power[hipass]

    return (freqs, power)

def boxsmooth(x, len, keepends=True):
    if len % 2 == 0:
        len += 1
    window = np.ones(len, 'd')
    smoothed = np.convolve(window/window.sum(), x, mode='valid')
    if keepends:
        xs = x.copy()
        halflen = (len-1)/2
        xs[halflen:-halflen] = smoothed
        smoothed = xs
    return smoothed

def local_func(f, t, x, w):
    """Compute a function over a window for each datum in a time series"""
    x_func = np.zeros_like(t, dtype='f')
    for i, jd in enumerate(t.jd):
        sel = (t.jd >= (jd - w)) & (t.jd <= (jd + w))
        x_func[i] = f(x[sel])
    return x_func

def running_func(f, t, x, w, lims=None):
    """Compute a function over a window for each day in a time segment"""
    if lims is None:
        lims = np.floor((t.jd.min() + w, t.jd.max() - w))
    t_func = np.arange(lims[0], lims[1], dtype='i')
    x_func = np.zeros_like(t_func, dtype='f')
    for i, jd in enumerate(t_func):
        sel = (t.jd >= (jd - w)) & (t.jd <= (jd + w))
        if np.sum(sel) == 0:
            x_func[i] = np.nan
        else:
            x_func[i] = f(x[sel])
    t_func = astropy.time.Time(t_func, format='jd')
    return t_func, x_func

def decimal_hour(t):
    """Return the hour of the day for every time in t"""
    datetime = t.datetime
    result = []
    for d in datetime:
        hour = d.hour
        minute = d.minute
        second = d.microsecond / 1e6
        dec_hour = hour + (minute/60.) + (second/3600.)
        result.append(dec_hour)
    return np.array(result)

def lag_indices(t, dt):
    # Find the indices of t such that t + dt exists in t,
    # with t quantized in days
    jds = t.jd
    # Center measurement time distribution to middle of julian day to
    # minimize problems with observations on bin edges
    fracs = jds - jds.astype('i')
    jds += (0.5 - np.median(fracs))
    # Quantize to integer days ranging from 0 to maxday
    days = np.floor(jds).astype('i')
    minjd = days.min()
    days = days - minjd
    maxday = days.max()
    # Build new evenly sampled array of days from 0 to maxday
    # and a boolean array indicating which of those days have elements
    daylist = np.arange(maxday+1)
    setdays = np.zeros(maxday+1, dtype='bool')
    setdays[days] = True
    # Build a list of indices for days which have measurements
    ixmap = np.zeros(maxday+1)
    ixmap[:] = np.nan
    ixmap[setdays] = np.arange(setdays.sum())
    # Build a boolean array indicating which days also have a
    # measurement dt days in the future
    match = np.zeros(maxday+1, dtype='bool')
    match[0:-dt] = np.logical_and(setdays[0:-dt], setdays[dt:])
    # Build a list of indices of t with (1) days for which a
    # measurements exists dt days in the future (2) the index of the
    # corresponding future measurement in (1).
    ix1 = ixmap[daylist[match]].astype('i')
    ix2 = ixmap[(daylist + dt)[match]].astype('i')
    return ix1, ix2

def lag_correlate(t, dt, x, method='Pearson'):
    ix1, ix2 = lag_indices(t, dt)
    N = ix1.size
    if method == 'Pearson':
        r, p = scipy.stats.pearsonr(x[ix1], x[ix2])
    elif method == 'Spearman':
        r, p = scipy.stats.spearmanr(x[ix1], x[ix2])
    else:
        raise Exception("Unknown method '%s'" % method)
    return r, p, N

def uneven_acf(t, x, Ndays=None, method='Pearson', lags=None):
    if lags is None and Ndays is not None:
        lags = np.arange(Ndays, dtype='i') + 1
    coeffs = np.zeros(lags.size)
    matches = np.zeros(lags.size)
    for i, lag in enumerate(lags):
        r, p, N = lag_correlate(t, lag, x, method=method)
        coeffs[i] = r
        matches[i] = N
    return lags, coeffs, matches

def plot_lag_correlation(t, dt, x):
    ix1, ix2 = lag_indices(t, dt)
    N = ix1.size
    Pr, p = scipy.stats.pearsonr(x[ix1], x[ix2])
    Sr, p = scipy.stats.spearmanr(x[ix1], x[ix2])

    plt.figure()
    plt.plot(x[ix1], x[ix2], 'ko')
    plt.xlabel("Day i")
    plt.ylabel("Day i + %i" % dt)
    plt.title("%i-day correlation (N=%i, Pr=%0.3f, Sr=%0.3f)" % (dt, N, Pr, Sr))

def plot_uneven_acf(t, x, Ndays=None, method='Pearson', lags=None,
                    title=None, unit='d', peak=True, minpeak=0., smooth=False, fullrange=False):
    lags, coeffs, matches = uneven_acf(t, x, Ndays, method, lags)
    if smooth:
        smoothed = boxsmooth(coeffs, smooth)
        minsmooth = (smooth - 1)/2 + 1
    if peak:
        if smooth: x = smoothed
        else:      x = coeffs
        acf_peaks = peaks(lags, x)
        if smooth and minpeak < minsmooth:
            minpeak = minsmooth
        acf_peaks = acf_peaks[acf_peaks >= minpeak]
    if unit == 'yr':
        xscale = 365.25
        xlabel = "Lag [yr]"
    else:
        xscale = 1.
        xlabel = "Lag [d]"
    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.plot(lags/xscale, matches, 'ko')
    plt.xlabel(xlabel)
    plt.ylabel("Matches")
    if title is not None:
        plt.title(title)
    plt.subplot(2,1,2)
    plt.plot(lags/xscale, coeffs, 'ko-')
    #plt.axhline(0, color='k')
    if fullrange:
        plt.ylim(-1,+1)
    if smooth:
        plt.plot(lags/xscale, smoothed, 'r-')
    if peak:
        plt.axvline(acf_peaks[0]/xscale, color='r')
        ax = plt.gca()
        trans = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        plt.text(acf_peaks[0]/xscale + 1, 0.9, "%0.1f" % (acf_peaks[0]/xscale), color='r', transform=trans)
    plt.xlabel(xlabel)
    plt.ylabel("Correlation Coeff")
    return lags, coeffs, matches

def append(t1, x1, t2, x2, e1=None, e2=None):
    # t.decimalyear fails if array is zero length
    if t1.size == 0:
        t1decyr = []
    else:
        t1decyr = t1.decimalyear
    if t2.size == 0:
        t2decyr = []
    else:
        t2decyr = t2.decimalyear
    t3 = np.append(t1decyr, t2decyr)
    ixsort = np.argsort(t3)
    if t3.size == 0:
        # unable to init zero-length array like Time([], format='decimalyear')
        # this reuses given zero-length array
        t3 = t1
    else:
        t3 = astropy.time.Time(t3, format='decimalyear')
    x3 = np.append(x1, x2)
    if e1 is None:
        return t3[ixsort], x3[ixsort]
    else:
        e3 = np.append(e1, e2)
        return t3[ixsort], x3[ixsort], e3[ixsort]
