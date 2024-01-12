import numpy as np
from scipy.interpolate import interp1d

def interp1flat(x, y, x2, method):
    # Sort x and y based on x values
    inds = np.argsort(x)
    sx = np.sort(x)

    # Pad the sorted x values for flat extrapolation
    sx_padded = np.concatenate(([min(np.min(x), np.min(x2)) - 1], sx, [max(np.max(x), np.max(x2)) + 1]))

    # Reorder and pad y values correspondingly
    sy = y[inds]
    sy_padded = np.concatenate(([sy[0]], sy, [sy[-1]]))

    # Interpolate with the specified method and flat extrapolation
    f = interp1d(sx_padded, sy_padded, kind=method, fill_value="extrapolate")

    # Perform the interpolation/extrapolation
    y2 = f(x2)
    return y2
