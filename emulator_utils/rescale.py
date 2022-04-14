"""
rescale.py
==============

Rescaling functions

"""
import numpy as np

def rescaleMinMax(f):
    """
    Scale values to a unitary range

    Params
    ------
    f: ndarray(ndarray(float))
       input values to rescale. Expect size of num_evals x ell_values 
    Returns
    -------
    fmin: ndarray(float)
       minimum f values for range (size of ell_values)
    fmax: ndarray(float)
       maximum f values for range (size of ell_values)
    fscaled: ndarray(ndarray(float))
       scaled f values (0 for minimum value, 1 for maximum value)

    """

    fmin = np.min(f,axis=0)
    fmax = np.max(f,axis=0)
    fscaled = (f - fmin) / (fmax - fmin)
    return fmin, fmax, fscaled


def scaleMinMax(fmin, fmax, f):
    '''
    Scale values to a unitary range, given known bounds
    Params
    ------
    fmin: ndarray(float)
       minimum f values (size of ell_values)
    fmax: ndarray(float)
       maximum f values (size of ell_values)
    f: ndarray(ndarray(float))
       input values to rescale. Expect size of num_evals x ell_values
    Returns
    -------
    fscaled: ndarray(ndarray(float))
       scaled f values (0 for minimum value, 1 for maximum value)
    '''
    return (f - fmin) / (fmax - fmin)


def unscaleMinMax(fmin, fmax, f):
    '''
    Undo scaling of values to a unitary range, given original bounds
    Params
    ------
    fmin: ndarray(float)
       minimum f values (size of ell_values)
    fmax: ndarray(float)
       maximum f values (size of ell_values)
    f: ndarray(ndarray(float))
       scaled values to unscale. Expect size of num_evals x ell_values
    Returns
    -------
    funscaled: ndarray(ndarray(float))
       unscaled f values
    '''

    return (f*(fmax - fmin)) + fmin


def rescale01(f):
    '''
    Scale values to a mean of zero and standard deviation of 1
    Params
    ------
    f: ndarray(ndarray(float))
       input values to rescale. Expect size of num_evals x ell_values
    Returns
    -------
    fmean: ndarray(float)
       mean f values (size of ell_values)
    fmax: ndarray(float)
       standard deviation of f values  (size of ell_values)
    fscaled: ndarray(ndarray(float))
       scaled f values
    '''
    fmean = np.mean(f,axis=0)
    fstd = np.std(f,axis=0)
    fscale = (f-fmean)/fstd
    return fmean, fstd, fscale

def scale01(fmean, fstd, f):
    '''
    Scale values to a mean of zero and standard deviation of 1, given known mean and std
    Params
    ------
    f: ndarray(ndarray(float))
       input values to rescale. Expect size of num_evals x ell_values
    fmean: ndarray(float)
       mean f values (size of ell_values)
    fmax: ndarray(float)
       standard deviation of f values  (size of ell_values)
    Returns
    -------
    fscaled: ndarray(ndarray(float))
       scaled f values
    '''

    return (f - fmean)/fstd

def unscale(fmean, fstd, f):
    '''
    Unscale values to recover mean and standard deviation given prior scaling 
    Params
    ------
    f: ndarray(ndarray(float))
       input values to unscale. Expect size of num_evals x ell_values
    fmean: ndarray(float)
       original mean f values (size of ell_values)
    fmax: ndarray(float)
       original standard deviation of f values  (size of ell_values)
    Returns
    -------
    funscaled: ndarray(ndarray(float))
       unscaled f values
    '''

    return (f*fstd) + fmean
