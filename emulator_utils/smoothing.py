"""
smoothing.py
============
Smoothing routines

"""
import numpy as np
import scipy.signal


__all__ = ("savgol", "gaussian", )

def savgol(data1d_array):
    """
    savgol filter
    
    Parameters
    ----------
    data1d_array: float
        input data

    Returns
    -------
    out1d: float
        output_data

    """
    out1d = scipy.signal.savgol_filter(data1d_array, 35, 7)
    return out1d

def gaussian(data1d_array):
    """
    Gaussian smmoothing. Currently not implemented
    """
    return NotImplemented

