import numpy as np
import scipy.signal


__all__ = ("savgol", "gaussian", )

def savgol(data1d_array):
    out1d = scipy.signal.savgol_filter(data1d_array, 35, 7)
    return out1d

def gaussian(data1d_array):
    return NotImplemented

