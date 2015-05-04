import numpy as np
from matplotlib import pyplot as plt
import scipy.signal
import scipy.optimize
import astropy.time
import timeit

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

def find_coincident(t1, t2, dt, unit='d'):
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

def season_indices(t):
    """
    Return indice array for which maps time sample to a season

    Example:
    >>> seasons = season_indexes(t)
    >>> for i in range(len(t)):
    >>>     print "Time point %i at %s is in season %i" % (i, t[i], seasons[i])
    """
    edges = season_edges(t)
    season_ixs = np.digitize(t.jyear, edges.jyear)
    N = season_ixs.size
    seasons = []
    for ix in np.unique(season_ixs):
        season_members = np.arange(N)[season_ixs == ix] # bool array to index array
        seasons.append(season_members)
    return seasons

def seasonal_series(t, y):
    """
    Return array of time series for each season.
    
    Example:
    >>> ts, ys = seasonal_series(t, y)
    >>> for i in range(len(ts)):
    >>>     print "Season %i has time samples %s and data %s" % (i, ts[i], ys[i])
    """
    season_ixs = season_indices(t)
    ts = []
    ys = []
    for season in season_ixs:
        ts.append(astropy.time.Time(t.jyear[season], format='jyear', scale=t.scale))
        ys.append(y[season])
    return ts, ys

def seasonal_means(t, y):
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
    ts, ys = seasonal_series(t, y)
    t_means = [t.jyear.mean() for t in ts]
    t_means = astropy.time.Time(t_means, format='jyear', scale=t.scale)
    y_means = np.array([y.mean() for y in ys])
    y_std = np.array([y.std() for y in ys])
    y_N = np.array([y.size for y in ys])
    return t_means, y_means, y_std, y_N

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
        peaks = find_peaks(power, thresh=thresh)
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

def residual_lombscargle(t, y, P, **kwargs):
    """Lomb-Scargle periodogram with period P removed from time series"""
    sine = find_sine(t.jyear, y, P)
    ynew = y - sine
    return lombscargle(t, ynew, **kwargs)

def samesamp_lombscargle(t, y, P, **kwargs):
    """Lomb-Scargle periodogram of sine of period P with same sampling as time series"""
    sine = find_sine(t.jyear, y, P)
    return lombscargle(t, sine, **kwargs)

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
