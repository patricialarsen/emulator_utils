"""
read_data.py
==============
Functions to read hacc outputs

"""

import numpy as np

def readpowerspec(path):
    """
    Read in power spectrum from a file path

    Parameters
    ----------
    path: float
       absolute path of power spectrum file

    Returns
    -------
    k: ndarray(float)
        wavenumbers in units of h/Mpc
    Pk: ndarray(float)
        power spectrum in units of (Mpc/h)^3
    Pkerr: ndarray(float)
        power spectrum error bars in units of (Mpc/h)^3
    nmodes: ndarray(float)
        number of modes for power spectrum calculation in units of (Mpc/h)^3

    """
    file_in = np.loadtxt(path, skiprows=1)
    return file_in[:,0], file_in[:,1], file_in[:,2], file_in[:,3]

def readcorr(path):
    """
    Read in correlation function from a file path

    Parameters
    ----------
    path: float
       absolute path of correlation function file

    Returns
    -------
    rlo: ndarray(float)
        minimum bin distance in units of Mpc/h
    rhi: ndarray(float)
        maximum bin distance in units of Mpc/h
    corr: ndarray(float)
        correlation function
    count: ndarray(int)
        global count 
    binsum: ndarray(int)
        global bin sum 

    """
    file_in = np.loadtxt(path, skiprows=1)
    return file_in[:,0], file_in[:,1], file_in[:,2], file_in[:,3], file_in[:,5]
