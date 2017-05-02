import numpy as np

def hathaway1994_cycle(t, A=193., b=54., c=0.8, t0=-4.):
    """Hathaway 1994 solar cycle model; Planck-like function"""
    x = (t-t0)/b
    return A * (x)**3 / (np.exp(x**2) - c)

def hathaway1994_cycle_offset(t, A=193., b=54., c=0.8, t0=-4., A0=0.):
    """Hathaway 1994 solar cycle model with nonzero offset"""
    return A0 + hathaway1994_cycle(t, A, b, c, t0)

def hathaway1994_cycle_constrained(popt):
    """Hathawy 1994 solar cycle model with fixed (b, c, t0)"""
    return lambda t, A, A0: hathaway1994_cycle_offset(t.decimalyear, A, popt[1], popt[2], popt[3], A0)

def du2011_cycle(t, A=109.0, B=25.2, alpha=4.87e-3, t_m=52.3):
    """Du 2011 solar cycle model; a skewed Gaussian"""
    t_diff = t - t_m
    return A * np.exp(-t_diff**2/(2 * (B*(1 + alpha*t_diff))**2))

def du2011_cycle_offset(t, A=109.0, B=25.2, alpha=4.87e-3, t_m=52.3, A0=0.):
    """Du 2011 solar cycle model with nonzero offset"""
    return A0 + du2011_cycle(t, A, B, alpha, t_m)

def du2011_cycle_constrained(popt):
    """Du 2011 solar cycle model with fixed (B, alpha, t_m)"""
    return lambda t, A, A0: du2011_cycle_offset(t.decimalyear, A, popt[1], popt[2], popt[3], A0)

def du2011_alpha(t0, t_m, unit='year'):
    """Du 2011 empirical alpha as a function of t_m"""
    if unit == 'year':
        t_mm = (t_m - t0)*12. # convert years to months; offset by t0
    elif unit == 'month':
        t_mm = t_m
    alpha = (48.4 - 1.35*t_mm + 0.00943*t_mm**2)/1e3
    return alpha

def du2011_cycle_3param_offset(t, A=109.0, B=25.2, t_m=52.3, A0=0., unit='year'):
    """Du 2011 solar cycle model; 3 parameter version"""
    alpha = du2011_alpha(t[0], t_m, unit=unit)
    return du2011_cycle_offset(t, A, B, alpha, t_m, A0)

def du2011_B(t0, t_m, unit='year'):
    """Du 2011 empirical B as a function of t_m"""
    if unit == 'year':
        t_mm = (t_m - t0)*12. # convert years to months; offset by t0
        #print "XXX: t0=%0.3f t_m=%0.3f t_mm=%0.3f" % (t0, t_m, t_mm)
    elif unit == 'month':
        t_mm = t_m
    B = -22.0 + 1.53*t_mm - 0.0115*t_mm**2 # [months]
    if unit == 'year':
        B = B/12. # convert months into years
    return B

def du2011_cycle_2param_offset(t, A=109.0, t_m=52.3, A0=0., unit='year'):
    """Du 2011 solar cycle model; 2 parameter version"""
    alpha = du2011_alpha(t[0], t_m, unit=unit)
    B = du2011_B(t[0], t_m, unit=unit)
    #print "XXX: B=%0.3e alpha=%0.3e" % (B, alpha)
    return du2011_cycle_offset(t, A, B, alpha, t_m, A0)
