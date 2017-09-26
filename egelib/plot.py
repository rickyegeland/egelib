import numpy as np
from matplotlib import pyplot as plt

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
