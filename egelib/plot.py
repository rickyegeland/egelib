import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

def square_range(x, y=None, identity=True, margin=0.1):
    """Make the x range equal to the y range and plot an identity line """
    if y is None: y = x
    if x.size == 0 or x.size != y.size: return
    combined = np.append(x, y)
    range = np.ptp(combined)
    min = combined.min() - range*margin
    max = combined.max() + range*margin
    plt.xlim(min, max)
    plt.ylim(min, max)
    if identity:
        linex = np.linspace(min, max, 3)
        liney = linex
        plt.plot(linex, liney, 'k-', label='y = x')

def set_ticks(majmult, minmult, majlen=10, minlen=5, size=18, axis='y'):
    if axis == 'y':
        ax = plt.gca().yaxis
    elif axis == 'x':
        ax = plt.gca().xaxis
    elif isinstance(axis, mpl.axis.Axis):
        ax = axis
    else:
        raise Exception('axis not specified correctly')
    ax.set_major_locator(mpl.ticker.MultipleLocator(majmult))
    ax.set_minor_locator(mpl.ticker.MultipleLocator(minmult))
    ax.set_tick_params(which='major', direction='in', labelsize=size, length=majlen)
    ax.set_tick_params(which='minor', direction='in', length=minlen)

def logerrorbar(x, y, yerr=None, xerr=None, logx=False, logy=False, ax=None, **kwargs):
    """Plot an errorbar plot with values and errors transformed to log10"""
    if ax is None:
        ax = plt.gca()
    if logx:
        logx = np.log10(x)
        logxp = np.log10(x + xerr) - logx
        logxm = logx - np.log10(x - xerr)
        x = logx
        xerr = np.array([logxp, logxm])
    if logy:
        logy = np.log10(y)
        logyp = np.log10(y + yerr) - logy
        logym = logy - np.log10(y - yerr)
        y = logy
        yerr = np.array([logyp, logym])
    ax.errorbar(x, y, yerr, xerr, **kwargs)
