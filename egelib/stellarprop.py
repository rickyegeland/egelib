def noyes84_logRpHK(S, BmV):
    """Return R prime HK, the color-corrected activity index from Noyes et al. 1984"""
    log_C_cf = 1.13 * BmV**3 - 3.91 * BmV**2 + 2.84 * BmV - 0.47
    R_HK = 1.340E-4 * 10**log_C_cf * S
    log_R_phot = -4.898 + 1.918 * BmV**2 - 2.893 * BmV**3
    Rp_HK = R_HK - 10**log_R_phot
    log_Rp_HK = np.log10(Rp_HK)
    return log_Rp_HK

def noyes84_tau_c(BmV):
    """Return tau_c, the turbulent convective turnover time [days] from Noyes et al. 1984"""
    x = 1 - BmV
    if x > 0:
        log_tau_c = 1.362 - 0.166*x + 0.025*x**2 - 5.323*x**3
    else:
        log_tau_c = 1.362 - 0.14*x
    return 10**log_tau_c # units: days

def noyes84_rossby(P, BmV):
    """Return the Rossby number = P[days]/tau_c, using tau_c from Noyes et al. 1984"""
    tau_c = noyes84_tau_c(BmV)
    return P/tau_c

def saar99_tau_c_E(BmV):
    """Return tau_c, the turbulent convective turnover time [days] from Saar & Brandenburg 1999"""
    if BmV < 1:
        tau_c = -3.3300 + 15.382*BmV - 20.063*BmV**2 + 12.540*BmV**3 - 3.1466*BmV**4
    else:
        tau_c = 25.
    return tau_c

def saar99_rossby_empirical(P, BmV):
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
