import numpy as np

# TODO: remove *_error() in favor of design in which:
# y_func(x, e_x=None) returns y, e_y when e_x is not None

# CONSTANTS
#   Source: IAU 2015 Resolution B2, B3
L0 = 3.0128e28 # W
LSUN = 3.828e26 # W
LSUN_cgs = LSUN * 1e7 # erg s^-1
TSUN = 5772. # K
RSUN = 6.957e8 # m
MBOL_SUN = -2.5 * np.log10(LSUN/L0)
FBOL_SUN = LSUN / (4 * np.pi * RSUN**2) # W m^-2
GMSUN = 1.3271244e20 # m^3 s^-2
AU = 149597870700. # m, IAU Resolution 1 (1976)
MAS2RAD = 1000. / 3600. * np.pi / 180. # 1 rad / 1 mas


def noyes84_logRpHK(S, BmV):
    """Return R prime HK, the color-corrected activity index

    Ref: Noyes et al. 1984
    """
    S = np.asarray(S)
    BmV = np.asarray(BmV)
    logC_cf = 1.13 * BmV**3 - 3.91 * BmV**2 + 2.84 * BmV - 0.47
    DlogC_cf = np.zeros_like(BmV) # for correction of "nonphysical maximum"
    x = (0.63 - BmV)
    sel = BmV < 0.63
    DlogC_cf[sel] = 0.135*x[sel] - 0.814*x[sel]**2 + 6.03*x[sel]**3
    logC_cf += DlogC_cf
    R_HK = 1.340E-4 * 10**logC_cf * S
    logR_phot = -4.898 + 1.918 * BmV**2 - 2.893 * BmV**3
    Rp_HK = R_HK - 10**logR_phot
    logRp_HK = np.log10(Rp_HK)
    return logRp_HK

def noyes84_logRpHK_to_S(logRpHK, BmV):
    """Return Mount Wilson S-index given logRpHK and color index

    Ref: Noyes et al. 1984
    """
    logRpHK = np.asarray(logRpHK)
    BmV = np.asarray(BmV)
    logC_cf = 1.13 * BmV**3 - 3.91 * BmV**2 + 2.84 * BmV - 0.47
    DlogC_cf = np.zeros_like(BmV) # for correction of "nonphysical maximum"
    x = (0.63 - BmV)
    sel = BmV < 0.63
    DlogC_cf[sel] = 0.135*x[sel] - 0.814*x[sel]**2 + 6.03*x[sel]**3
    logC_cf += DlogC_cf
    logR_phot = -4.898 + 1.918 * BmV**2 - 2.893 * BmV**3
    Rp_HK = 10**logRpHK
    R_HK = Rp_HK + 10**logR_phot
    S = R_HK/(1.340E-4 * 10**logC_cf)
    return S

def noyes84_logRpHK_error(S, e_S, BmV, e_BmV):
    """Uncertainty in logRpHK"""
    ln10 = np.log(10)
    var_BmV = e_BmV**2
    var_S = e_S**2
    # variance in 10**logC
    # warning: ignores DlogC_cf correction
    logC = 1.13 * BmV**3 - 3.91 * BmV**2 + 2.84 * BmV - 0.47
    var_logC = (1.13 * 3 * BmV**2 - 3.91 * 2 * BmV + 2.84 )**2 * var_BmV
    var_10logC = (10**logC * ln10)**2 * var_logC
    # variance R_HK
    a = 1.340E-4
    R_HK = a * 10**logC * S
    var_R_HK = (a * S)**2 * var_10logC + (a * 10**logC)**2 * var_S
    # variance 10**logRphot
    logR_phot = -4.898 + 1.918 * BmV**2 - 2.893 * BmV**3
    var_logR_phot = (1.918 * 2 * BmV - 2.893 * 3 * BmV**2)**2 * var_BmV
    var_10logR_phot = (10**logR_phot * ln10)**2 * var_logR_phot
    # variance RpHK
    RpHK = R_HK - 10**logR_phot
    var_RpHK = var_R_HK + var_10logR_phot
    # variance logRpHK
    var_logRpHK = (RpHK * ln10) ** -2 * var_RpHK
    # result as standard deviation
    return np.sqrt(var_logRpHK)

def noyes84_tau_c(BmV, e_BmV=None):
    """Return tau_c, the turbulent convective turnover time [days]

    Ref: Noyes et al. 1984"""
    BmV = np.asarray(BmV)
    log_tau_c = np.zeros_like(BmV)
    x = 1 - BmV
    blue = x > 0
    red = np.logical_not(blue)
    log_tau_c[blue] = 1.362 - 0.166*x[blue] + 0.025*x[blue]**2 - 5.323*x[blue]**3
    log_tau_c[red] = 1.362 - 0.14*x[red]
    tau_c = 10**log_tau_c # [days]
    if e_BmV is not None:
        e_BmV = np.asarray(e_BmV)
        if e_BmV.size == 1:
            e_BmV = np.ones_like(BmV) * e_BmV
        e_log_tau_c = np.zeros_like(e_BmV)
        e_log_tau_c[blue] = e_BmV[blue] * np.sqrt((-0.166 + 2*0.025*x[blue] - 3*5.323*x[blue]**2)**2)
        e_log_tau_c[red] = e_BmV[red] * 0.14
        e_tau_c = np.abs(tau_c * np.log(10) * e_log_tau_c)
        return tau_c, e_tau_c
    else:
        return tau_c

def noyes84_rossby(P, BmV, e_P=None, e_BmV=None):
    """Return the Rossby number = P[days]/tau_c, using tau_c
    
    Ref: Noyes et al. 1984"""
    if e_P is None:
        tau_c = noyes84_tau_c(BmV)
        Ro = P/tau_c
        return Ro
    else:
        tau_c, e_tau_c = noyes84_tau_c(BmV, e_BmV)
        Ro = P/tau_c
        e_Ro = np.sqrt( Ro**2 * (e_P/P)**2 + (e_tau_c/tau_c)**2)
        return Ro, e_Ro

def noyes84_rossby_activity(logRpHK):
    """Return the Rossby number given the activity
    
    Ref: Noyes et al. 1984 eq (3)"""
    y = 5 + logRpHK
    logRo = 0.324 - 0.400*y - 0.283 * y**2 - 1.325 * y**3
    return 10**logRo

def bv2teff_noyes84(BmV):
    """Rough conversion from B-V to T_eff

    Ref: Noyes et al. 1984
      "fits Johnson (1966) data within 0.002 dex for 0.4 < BmV < 1.4"
    """
    logTeff = 3.908 - 0.234*BmV
    return 10 ** logTeff

def teff2bv_noyes84(Teff):
    """Rough conversion from Teff to B-V
    
    Ref: Noyes et al. 1984
      "fits Johnson (1966) data within 0.002 dex for 0.4 < BmV < 1.4"
    """
    logTeff = np.log10(Teff)
    BmV = (3.908 - logTeff) / 0.234
    return BmV

def saar99_tau_c_E(BmV):
    """Return tau_c, the turbulent convective turnover time [days]

    Ref: Saar & Brandenburg 1999
    """
    BmV = np.asarray(BmV)
    tau_c = np.zeros_like(BmV)
    sel = BmV < 1
    notsel = np.logical_not(sel)
    tau_c[sel] = -3.3300 + 15.382*BmV[sel] - 20.063*BmV[sel]**2 + 12.540*BmV[sel]**3 - 3.1466*BmV[sel]**4
    tau_c[notsel] = 25.
    return tau_c

def saar99_rossby_empirical(P, BmV):
    """Return the Rossby number = P[days]/tau_c, using tau_c

    Ref: Saar & Brandenburg 1999
    """
    tau_c = saar99_tau_c_E(BmV)
    return P/(4.*np.pi*tau_c)

# Barnes 2007; altered Skumanich
def barnes07_age_Prot(Prot, BmV, n=0.5189, a=0.7725, b=0.601, c=0.40):
    """Returns age (Gyr) as a function of rotation period and B-V color

    Ref: Barnes 2007 eq. (3)
    """
    # TODO: uncertainties
    logt = (1./n) * (np.log10(Prot) - np.log10(a) - b * np.log10(BmV - c))
    t = 10 ** logt # Myr
    return t / 1000. # Gyr

def barnes07_Prot_age(t, BmV, n=0.5189, a=0.7725, b=0.601, c=0.40, unit='Gyr'):
    """Returns Prot (days) as a function of age and B-V color

    Ref: Barnes 2007 eq. (3)

    unit := units of age (Gyr (default), yr, or Myr)
    """
    # TODO: uncertainties
    if unit == 'Gyr':
        t = t * 1000. # Gyr -> Myr
    elif unit == 'yr':
        t = t / 1e6 # yr -> Myr
    elif unit == 'Myr':
        pass
    else:
        raise Exception("Unsupported time unit '%s'" % unit)
    # Expects t in Myr
    logP = n * np.log10(t) + b * np.log10(BmV - c) + np.log10(a)
    return 10 ** logP # days

def t_chromo_duncan(RpHK, a=10.725, b=-1.334, c=0.4085, d=-0.0522):
    R5 = 1e5 * RpHK
    logt = a + b * R5 + c * R5**2 + d * R5**3
    return (10 ** logt) / 1e6 # yr -> Myr

def t_chromo_soderblom(logRpHK, a=-1.50, b=2.25):
    logt = a*logRpHK + b
    return 10 ** logt / 1e6 # yr -> Myr

# Guinan & __ 2008 Fig 1
def t_gyro_guinan(P, y0=1.865, a=-2.854, b=0.08254):
    return ( -(np.log10(P) + y0) / a )**(1./b)

def mamajek08_logRpHK_edge():
    """The dividing point between Mamajek et al. 2008's "active" and "very active" samples

    Mamjek says the edge is -4.3, but in their Fig 7 their curves
    clearly intersect at a lesser value.  Using -4.3 results in a
    step discontinuity.  Here, we calculate the Ro where the curves
    intersect, the result is .

    logRpHK_edge = (-(0.808 - 0.233) - (0.689*4.23 - 2.966*4.52)) / (0.689 - 2.966)
    -4.355226174791392
    """
    return -4.355226174791392

def mamajek08_logRpHK_max():
    """The maximum valid value of logRpHK using Mamajek's relations

    From setting Mamajek et al. 2008 eq 7 = 0.
    logRpHK_min = (0.233/0.689) - 4.23
    -3.8918287373004357
    """
    return -3.8918287373004357

def mamajek08_Ro_logRpHK(logRpHK, extrapolate=False):
    """Mamajek 2008 Rossby number as a function of activity"""
    # Mamjek relations only valid up to logRpHK = -5.0
    # Use eq (5) to find the corresponding max Ro (~2.23)
    logRpHK = np.asarray(logRpHK)
    scalar_input = False
    if logRpHK.ndim == 0:
        logRpHK = logRpHK[None]
        scalar_input = True
    logRpHK_edge = mamajek08_logRpHK_edge()
    active = (logRpHK >= -5.0) & (logRpHK <= logRpHK_edge)
    very_active = logRpHK > logRpHK_edge
    inactive = logRpHK < -5.0
    Ro = np.ones_like(logRpHK)
    Ro[active] = 0.808 - 2.966 * (logRpHK[active] + 4.52) # eq (5)
    Ro[very_active] = 0.233 - 0.689 * ( logRpHK[very_active] + 4.23) # eq (7)
    if extrapolate:
        Ro[inactive] = 0.808 - 2.966 * (logRpHK[inactive] + 4.52) # eq (5)
    else:
        Ro[inactive] = np.nan
    Ro[Ro < 0] = np.nan # Negative Ro is not valid
    if scalar_input:
        return np.squeeze(Ro)
    return Ro

def mamajek08_Prot_logRpHK(logRpHK, BmV, extrapolate=False):
    Ro = mamajek08_Ro_logRpHK(logRpHK, extrapolate=extrapolate)
    tau_c = noyes84_tau_c(BmV)
    return Ro * tau_c

def mamajek08_logRpHK_Ro_max():
    """The maximum Rossby number in Mamajek et al. 2008's empirical fit
    
    They cut log(RpHK) > -5.0, so we apply their equation (5) to that value.
    The result is ~ 2.23
    """
    return mamajek08_Ro_logRpHK(-5.0)

def mamajek08_logRpHK_Ro_edge():
    """The dividing point between Mamajek et al. 2008's "active" and "very active" samples

    Mamjek says the edge is 0.4, but in their Fig 7 their curves
    clearly intersect at a lesser value.  Using 0.4 results in a
    step discontinuity.  Here, we calculate the Ro where the curves
    intersect, the result is ~0.32.

    Ro_edge = ((4.522 - 4.23) + (1.451*0.233 - 0.337*0.814)) / (1.451 - 0.337)
    0.31935816876122064
    """
    Ro_edge = 0.31935816876122064
    return Ro_edge

def mamajek08_logRpHK_Ro(Ro, extrapolate=False):
    """Activity logR'_HK as a function of Rossby number

    Ref: Mamajek et al. 2008 equations (6) and (8)
    Note: Ro is from Noyes et al. 1984 (see noyes84_rossby())
          valid only for -5.0 < logRpHK < -4.3
    """
    # Mamjek relations only valid up to logRpHK = -5.0
    # Use eq (5) to find the corresponding max Ro (~2.23)
    Ro_max = mamajek08_logRpHK_Ro_max()
    Ro_edge = mamajek08_logRpHK_Ro_edge()
    Ro = np.asarray(Ro)
    scalar_input = False
    if Ro.ndim == 0:
        Ro = Ro[None]
        scalar_input = True
    # TODO: uncertainties!
    fast = Ro < Ro_edge
    mid = (Ro >= Ro_edge) & (Ro <= Ro_max)
    slow = Ro > Ro_max
    logRpHK = np.ones_like(Ro) * np.nan
    logRpHK[mid] =  -4.522 - 0.337 * (Ro[mid] - 0.814) # "normal" activity: equaiton (6)
    logRpHK[fast] = -4.23 - 1.451 * (Ro[fast] - 0.233) # very active: equation (8)
    if extrapolate:
        logRpHK[slow] = -4.522 - 0.337 * (Ro[slow] - 0.814)
    else:
        logRpHK[slow] = np.nan # Ro >= ~2.23 is undefined
    if scalar_input:
        return np.squeeze(logRpHK)
    return logRpHK
    
def mamajek08_logRpHK_age(t):
    """Activity logR'_HK as a function of age (in Gyr)

    Ref: Mamajek et al. 2008 equation (4)
    Note: valid for log(R'_HK) in [-4.0,-5.1] and log(t) in [6.7,9.9]
    """
    # TODO: uncertainties!
    logt = np.log10(t * 1e9) # log([yr])
    logRpHK = 8.94 - 4.849 * logt + 0.624 * logt**2 - 0.028 * logt**3
    return logRpHK

def mamajek08_age_logRpHK(logRpHK):
    """Age (in Gyr) as a function of activity log(R'_HK)

    Ref: Mamajek & Hillebrand 2008 equation (3)
    Note: valid for log(R'_HK) in [-4.0,-5.1] and log(t) in [6.7,9.9]
    """
    # TODO: uncertainties!
    logt = -38.053 - 17.912 * logRpHK - 1.6675 * logRpHK**2
    return 10**logt / 1e9 # Gyr

def mamajek08_logRpHK_logRx(logRx, e_logRx=None):
    """Activity log(R'_HK) as a function of activity log(R_X)

    Ref: Mamajek et al. 2008 equation (A1)
    Note: valid for log(R_X) in (-2.8, -7.4)
    """
    logRpHK =  -4.54 + 0.289 * (logRx + 4.92)
    if e_logRx is None:
        return logRpHK
    else:
        # logRpHK = p1 + p2 * (logRx + p3) = p1 + p2p3 + p2logRx
        p1 = -4.90
        e_p1 = 0.01
        p2 = 0.289
        e_p2 = 0.015
        p3 = 4.92
        p2p3 = p2 * p3
        e_p2p3 = p3 * e_p2
        p2logRx = p2 * logRx
        e_p2logRx = p2logRx * np.sqrt( (e_p2/p2)**2 + (e_logRx/logRx)**2)
        e_logRpHK = np.sqrt(e_p1**2 + e_p2p3**2 + e_p2logRx**2)
        return logRpHK, e_logRpHK

def mamajek08_Prot_age(t, BmV, unit='Gyr'):
    """Returns rotation period as a function of age and B-V color

    Ref: Mamajek & Hillebrand 2008 Table 10, equations 12--14
    Uses same function defined in Barnes 2007, revises the parameters
    """
    return barnes07_Prot_age(t, BmV, unit=unit, a=0.407, b=0.325, c=0.495, n=0.566)

def mamajek08_age_Prot(Prot, BmV):
    """Returns age as a function of rotation period and B-V color

    Ref: Mamajek & Hillebrand 2008 Table 10, equations 12--14
    Uses same function defined in Barnes 2007, revises the parameters
    """
    return barnes07_age_Prot(Prot, BmV, a=0.407, b=0.325, c=0.495, n=0.566)

def mamajek08_age2_logRpHK(logRpHK, BmV, extrapolate=False):
    """Age (in years) as a function of activity log(R'_HK) and B-V color

    Ref: Mamajek 2008 section 4.4.  First converts activity to rotation period,
         then rotation period to age.
    Note: valid for log(R'_HK) in [-4.0,-5.1] and log(t) in [6.7,9.9]
    """
    Prot = mamajek08_Prot_logRpHK(logRpHK, BmV, extrapolate=extrapolate)
    return mamajek08_age_Prot(Prot, BmV)

def differential_rotation(lat, A, B, C):
    """Return standard differential rotation profile \Omega = A + B*sin(lat)**2 + C*sin(lat)**4

    Input:
     - lat <float> : latitude in degrees
    \Omega Units depnd on units of coefficients.
    """
    
    lat_deg = lat *  np.pi/180.
    return A + B * np.sin(lat_deg)**2 + C * np.sin(lat_deg)**4

def omega_sun_snodgrass90(lat):
    """Sidereal solar surface rotation in deg/day as a function of latitude in degrees

    Based on tracking doppler features in the solar photosphere.

    Reference: Snodgrass and Ulrich 1990, ApJ
    """
    return differential_rotation(lat, 14.71, -2.39, -1.78)

def hall95_StoFlux(S, BmV, Teff=None):
    # Hall 1995 equation 8; Middelkoop 1982
    log_Ccf = -0.47 + 2.84 * BmV - 3.91 * BmV**2 + 1.13 * BmV**3
    Ccf = 10**log_Ccf
    if Teff is None:
        # Hall 1995 equation 11; Lang 1991
        logTeff = 3.923 - 0.247*BmV
        Teff = 10**logTeff
    # Hall 1995 equation 12; Middelkoop 1982
    F_HK = S * Ccf * Teff**4 * 1e-14
    return F_HK
    
def hall95_fluxToS(F_HK, BmV, Teff=None):
    # Hall 1995 equation 8; Middelkoop 1982
    log_Ccf = -0.47 + 2.84 * BmV - 3.91 * BmV**2 + 1.13 * BmV**3
    Ccf = 10**log_Ccf
    if Teff is None:
        # Hall 1995 equation 11; Lang 1991
        logTeff = 3.923 - 0.247*BmV
        Teff = 10**logTeff
    # Inverse of Hall 1995 equation 12; Middelkoop 1982
    S = F_HK / (Ccf * Teff**4 * 1e-14)
    return S

def hall1996_flux_cont_HK_bmy(bmy):
    # Hall 1996 Table 4
    if bmy >= -0.1 and bmy <= 0.41:
        logF = 8.179 - 2.887 * bmy
    elif bmy > 0.41 and bmy <= 0.8:
        logF = 8.906 - 4.657 * bmy
    else:
        raise Exception("b-y out of range")
    return 10**logF

def rutten1984_Ccf(BmV):
    # Rutten, R. G. M. 1984, A&A, 130, 353
    # Used in Hall, Lockwood, & Skiff 2007 (eq 5)
    logCcf = 0.25 * BmV**3 - 1.33 * BmV**2 + 0.43 * BmV + 0.24
    return 10**logCcf

def hall2007_S_relscale(bmy_A, BmV_A, Teff_A, K_A, bmy_B, BmV_B, Teff_B, K_B, factors=False):
    # Returns S_A/S_B: i.e., the relative scaling factor to convert S 
    #computed with B parameters to S computed with A parameters
    F_cont_A = hall1996_flux_cont_HK_bmy(bmy_A)
    F_cont_B = hall1996_flux_cont_HK_bmy(bmy_B)
    flux_factor = F_cont_A / F_cont_B
    Ccf_A = rutten1984_Ccf(BmV_A)
    Ccf_B = rutten1984_Ccf(BmV_B)
    Ccf_factor = Ccf_B / Ccf_A
    Teff_factor = (Teff_B / Teff_A)**4
    K_factor = K_B / K_A
    S_factor = flux_factor * Ccf_factor * Teff_factor * K_factor
    if factors:
        return (S_factor, flux_factor, Ccf_factor, Teff_factor, K_factor)
    else:
        return S_factor

def flower1996_BC_params(logT):
    # Corrections by Torres 2010 (2010AJ....140.1158T)
    # Do not use for M dwarfs; see Torres 2010 
    if logT > 3.90:
        params = [-0.118115450538963E+06, 0.137145973583929E+06, -0.636233812100225E+05,
                  0.147412923562646E+05, -0.170587278406872E+04, 0.788731721804990E+02]
    elif logT <= 3.90 and logT >= 3.70:
        params = [-0.370510203809015E+05, 0.385672629965804E+05, -0.150651486316025E+05,
                  0.261724637119416E+04, -0.170623810323864E+03, 0]
    elif logT < 3.70:
        params = [-0.190537291496456E+05, 0.155144866764412E+05, -0.421278819301717E+04, 
                  0.381476328422343E+03, 0, 0]
    else:
        raise Exception("Impossible Teff")
    return params

def flower1996_BC(Teff, e_Teff=0):
    # TODO: the vecorization of this is stupid.  Teff is allowed to be
    # a scalar or array, but there are way too many if statements in
    # this function to enable that.
    Teff = np.asarray(Teff)
    if Teff.size == 1:
        Teff = np.array([Teff])
    logT = np.log10(Teff)
    var_Teff = e_Teff**2
    var_logT = var_Teff / (Teff * np.log(10))**2
    if var_logT.size == 1:
        var_logT = np.ones_like(logT) * var_logT # expand array
    BC = np.zeros_like(logT)
    e_BC = np.zeros_like(logT)
    for i in range(logT.size):
        t = logT[i]
        if np.isnan(t):
            BC[i] = np.nan
            e_BC[i] = np.nan
            continue
        params = flower1996_BC_params(t)
        BC_poly = np.polynomial.polynomial.Polynomial(params)
        BC[i] = BC_poly(t)
        if e_Teff is not 0:
            # do the derivative of the Flower polynomial
            dBC = BC_poly.deriv()
            var_BC = var_logT[i] * dBC(t)**2
            e_BC[i] = np.sqrt(var_BC)
    if BC.size == 1:
        BC = BC[0]
        e_BC = e_BC[0]
    if e_Teff is not 0:
        return BC, e_BC
    else:
        return BC

def flower1996_BmV_to_Teff(BmV, e_BmV=0):
    # Corrections by Torres 2010 (2010AJ....140.1158T)
    # Do not use for M dwarfs; see Torres 2010 
    #
    # Polynomial for Main Sequence, Subgiants, & Giants
    # Do not use for supergiants
    params = [3.979145106714099, -0.654992268598245, 1.740690042385095, -4.608815154057166,
              6.792599779944473, -5.396909891322525, 2.192970376522490, -0.359495739295671]
    logTeff_poly = np.polynomial.polynomial.Polynomial(params)
    logTeff = logTeff_poly(BmV)
    Teff = 10**logTeff
    # TODO WARNING: assumes scalars; update to work with arrays (like above)
    if e_BmV is not 0:
        var_BmV = e_BmV**2
        dlogTeff = logTeff_poly.deriv()
        var_logTeff = var_BmV * dlogTeff(BmV)**2
        var_Teff = Teff**2 * np.log(10)**2 * var_logTeff
        e_Teff = np.sqrt(var_Teff)
        return Teff, e_Teff
    else:
        return Teff

def flower1996_BC_BmV(BmV, e_BmV=0):
    if e_BmV is 0:
        Teff = flower1996_BmV_to_Teff(BmV)
        return flower1996_BC(Teff)
    else:
        Teff, e_Teff = flower1996_BmV_to_Teff(BmV, e_BmV)
        return flower1996_BC(Teff, e_Teff)

# another constant
VMAG_SUN = MBOL_SUN - flower1996_BC(TSUN)

# def flower1996_BC_error(Teff, e_Teff):
#     var_Teff = e_Teff**2
#     logT = np.log10(Teff)
#     var_logT = var_Teff / (Teff * np.log(10))**2

#     # do the derivative of the Flower polynomial
#     params = flower1996_BC_params(logT)
#     BC = np.polynomial.polynomial.Polynomial(params)
#     dBC = BC.deriv()
#     var_BC = var_logT * dBC(logT)
#     return np.sqrt(var_BC)

def absolute_mag(Vmag, plx, e_Vmag=0, e_plx=0):
    VMag = Vmag + 5 * (1 + np.log10(plx))
    if e_Vmag is not 0 or e_plx is not 0:
        e_logplx = np.abs( e_plx / (plx * np.log(10)) )
        e_VMag = np.sqrt( e_Vmag**2 + (5 * e_logplx)**2 )
        return VMag, e_VMag
    else:
        return VMag

def bolometric_mag(VMag, BC, e_VMag=0, e_BC=0):
    Mbol = VMag + BC
    if e_VMag is not 0 or e_BC is not 0:
        e_Mbol = np.sqrt(e_VMag**2 + e_BC**2)
        return Mbol, e_Mbol
    else:
        return Mbol

def luminosity(Mbol, e_Mbol=0):
    L = (L0/LSUN) * 10 ** (-0.4 * Mbol)
    if e_Mbol is not 0:
        var_L = L**2 * (-0.4*np.log(10) * e_Mbol)**2
        e_L = np.sqrt(var_L)
        return L, e_L
    else:
        return L

def radius(L, T, e_L=0, e_T=0):
    R = np.sqrt(L) / T**2
    if e_L is not 0 or e_T is not 0:
        var_sqrtL = e_L**2/(4*L)
        var_Tsq = (2*T*e_T)**2
        var_R = R**2 * (var_sqrtL/L + var_Tsq/T**4) # warning: ignores - 2*var_AB/AB term
        e_R = np.sqrt(var_R)
        return R, e_R
    else:
        return R

def mass(logg, R, e_logg=0, e_R=0):
    # g = GM/R**2 => GM = g*R**2
    g = 10**logg / 100 # [cm/s^2] => [m/s^2]
    Rm = R * RSUN # [m]
    Rm2 = Rm**2 # [m^2]
    GM = g * Rm2 # [m^3 s^-2]
    M = GM/GMSUN
    if e_logg is not 0 or e_R is not 0:
        e_g = np.log(10) * g * e_logg / 100
        e_Rm = e_R * RSUN
        e_Rm2 = 2 * e_Rm * Rm # R always positive
        e_GM = GM * np.sqrt( (e_g/g)**2 + (e_Rm2/Rm2)**2 )
        e_M = e_GM / GMSUN
        return M, e_M
    else:
        return M
    
def Teff_to_T(Teff, e_Teff=0):
    T = Teff/TSUN
    if e_Teff is not 0:
        e_T = e_Teff/TSUN
        return T, e_T
    else:
        return T

def VMagTeff_to_LR(VMag, Teff, e_VMag=0, e_Teff=0, BC_func=flower1996_BC):
    if e_VMag is not 0 or e_Teff is not 0:
        T, e_T = Teff_to_T(Teff, e_Teff)
        BC, e_BC = BC_func(Teff, e_Teff)
        Mbol, e_Mbol = bolometric_mag(VMag, BC, e_VMag, e_BC)
        L, e_L = luminosity(Mbol, e_Mbol)
        R, e_R = radius(L, T, e_L, e_T)
        return L, R, e_L, e_R
    else:
        T = Teff_to_T(Teff)
        BC = BC_func(Teff)
        Mbol = bolometric_mag(VMag, BC)
        L = luminosity(Mbol)
        R = radius(L, T)
        return L, R

def vsiniR_to_Peq_sini(vsini, R, e_vsini=0, e_R=0):
    R_km = R * RSUN / 1000.
    Peq_sini = 2 * np.pi * R_km / vsini # [s]
    Peq_sini = Peq_sini / 3600. / 24. # [d]
    if e_vsini is not 0:
        e_Peq_sini = 2 * np.pi * Peq_sini * np.sqrt( (e_vsini/vsini)**2 + (e_R/R)**2 )
        return Peq_sini, e_Peq_sini
    else:
        return Peq_sini

def inclination_sini(vsini, R, Peq, e_vsini=0, e_R=0, e_Peq=0):
    if e_vsini is not 0:
        Peq_sini, e_Peq_sini = vsiniR_to_Peq_sini(vsini, R, e_vsini, e_R) # [d]
        sini = Peq / Peq_sini
        e_sini = np.sqrt( (e_Peq/Peq)**2 + (e_Peq_sini/Peq_sini)**2 )
        return sini, e_sini
    else:
        Peq_sini = vsiniR_to_Peq_sini(vsini, R) # [d]
        sini = Peq / Peq_sini
        return sini

def arcsin_lim(sini, e_sini=0):
    if np.abs(sini) > 1:
        if e_sini is not 0:
            return np.nan, np.nan
        else:
            return np.nan
    else:
        i = np.arcsin(sini) * (180./np.pi)
        if e_sini is not 0:
            e_i = e_sini/np.sqrt(1. - sini) * (180./np.pi)
            return i, e_i
        else:
            return i

def inclination(vsini, R, Peq, e_vsini=0, e_R=0, e_Peq=0):
    if e_vsini is not 0:
        sini, e_sini  = inclination_sini(vsini, R, Peq, e_vsini, e_R, e_Peq)
        inc, e_inc  = arcsin_lim(sini, e_sini)
        return inc, e_inc
    else:
        sini = inclination_sini(vsini, R, Peq)
        return arcsin_lim(sini)

def distance(plx):
    plx_rad = plx * MAS2RAD  # radians
    d_AU = 0.5 * 1/plx_rad # AU
    d_m = d_AU * AU # m
    return d_m

def angular_diam(R, plx):
    # R in Rsun units, plx in mas
    R_m = R * RSUN # meters
    d_m = distance(plx) # meters
    theta_rad = R_m / d_m # radians
    return theta_rad / MAS2RAD # mas

def wright2004_main_sequence(BmV):
    """Returns the function M_V(B-V) that defines the MS in Wright 2004"""
    
    coeffs = np.array([1.11255, 5.79062, -16.76829, 76.47777, -140.08488,
                       127.38044, -49.71805,  8.24265, 14.07945, -3.43155])[::-1]
    M_V = np.poly1d(coeffs)
    return M_V(BmV)

def wright2004_dVMag(BmV, VMag):
    """Returns Delta VMag, height above the main sequence"""
    return wright2004_main_sequence(BmV) - VMag

def wright2004_is_evolved(BmV, VMag):
    """Returns boolean array whether a star is evolved according to Wright 2004"""
    return wright2004_dVMag(BmV, VMag) >= 1.0
