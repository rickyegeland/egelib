import numpy as np

def noyes84_logRpHK(S, BmV):
    """Return R prime HK, the color-corrected activity index

    Ref: Noyes et al. 1984
    """
    logC_cf = 1.13 * BmV**3 - 3.91 * BmV**2 + 2.84 * BmV - 0.47
    R_HK = 1.340E-4 * 10**logC_cf * S
    logR_phot = -4.898 + 1.918 * BmV**2 - 2.893 * BmV**3
    Rp_HK = R_HK - 10**logR_phot
    logRp_HK = np.log10(Rp_HK)
    return logRp_HK

def noyes84_logRpHK_to_S(logRpHK, BmV):
    """Return Mount Wilson S-index given logRpHK and color index

    Ref: Noyes et al. 1984
    """
    logC_cf = 1.13 * BmV**3 - 3.91 * BmV**2 + 2.84 * BmV - 0.47
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

def noyes84_tau_c(BmV):
    """Return tau_c, the turbulent convective turnover time [days]

    Ref: Noyes et al. 1984"""
    log_tau_c = np.zeros_like(BmV)
    x = 1 - BmV
    sel = x > 0
    notsel = np.logical_not(sel)
    log_tau_c[sel] = 1.362 - 0.166*x[sel] + 0.025*x[sel]**2 - 5.323*x[sel]**3
    log_tau_c[notsel] = 1.362 - 0.14*x[notsel]
    return 10**log_tau_c # units: days

def noyes84_rossby(P, BmV):
    """Return the Rossby number = P[days]/tau_c, using tau_c
    
    Ref: Noyes et al. 1984"""
    tau_c = noyes84_tau_c(BmV)
    return P/tau_c

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
    if BmV < 1:
        tau_c = -3.3300 + 15.382*BmV - 20.063*BmV**2 + 12.540*BmV**3 - 3.1466*BmV**4
    else:
        tau_c = 25.
    return tau_c

def saar99_rossby_empirical(P, BmV):
    """Return the Rossby number = P[days]/tau_c, using tau_c

    Ref: Saar & Brandenburg 1999
    """
    tau_c = saar99_tau_c_E(BmV)
    return P/(4.*np.pi*tau_c)

# Barnes 2007; altered Skumanich
def t_gyro_barnes(P, BV, n=0.5189, a=0.7725, b=0.601):
    logt = (1./n) * (np.log10(P) - np.log10(a) - b * np.log10(BV - 0.4))
    return 10 ** logt # Myr

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

def differential_rotation(lat, A, B, C):
    """Return standard differential rotation profile \Omega = A + B*sin(lat)**2 + C*sin(lat)**4

    Input:
     - lat <float> : latitude in degrees
    \Omega Units depnd on units of coefficients.
    """
    lat *= np.pi/180.
    return A + B * np.sin(lat)**2 + C * np.sin(lat)**4

def omega_sun_snodgrass90(lat):
    """Solar surface rotation in deg/day as a function of latitude in degrees

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
