"""
precompute_quantities.py
========================
Pre-compute conserved quantities

"""

import numpy as np

def pk_ratio(k_vals,pk_vals,steps):
    """
    power spectrum ratio 

    Parameters
    ----------
    k_vals: ndarray(float)
        input wavenumbers
    pk_vals: ndarray(float)
        input power spectra
    steps: ndarray(int)
        simulation step list

    Returns
    -------
    ratio: ndarray(float)
        power spectrum ratio with respect to the final redshift

    """
    nsteps = len(steps)
    if len(k_vals)>1:
        assert((k_vals[0]==k_vals[1]).all())
    steps = np.array(steps).astype(int)
    base_step = np.max(steps)
    base_idx = np.where(steps==base_step)[0]
    pk_vals = np.array(pk_vals)
    return pk_vals/pk_vals[base_idx]


def corr_ratio(r_min, r_max, corr_vals, steps):
    """
    correlation function ratio

    Parameters
    ----------
    r_min: ndarray(float)
        input lower bin edges
    r_max: ndarray(float)
        input upper bin edges
    corr_vals: ndarray(float)
        input correlation functions
    steps: ndarray(int)
        simulation step list

    Returns
    -------
    ratio: ndarray(float)
        correlation function ratio with respect to the final redshift

    """
    nsteps = len(steps)
    if len(k_vals)>1:
        assert((r_min[0]==r_min[1]).all())
    steps = np.array(steps).astype(int)
    base_step = np.max(steps)
    base_idx = np.where(steps==base_step)[0]
    corr_vals = np.array(corr_vals)
    return corr_vals/corr_vals[base_idx]

def lin_ratio(k_vals,pk_vals,steps,params):
    """
    ratio with respect to linear theory

    Parameters
    ----------
    k_vals: ndarray(float)
        input wavenumbers
    pk_vals: ndarray(float)
        input power spectra
    steps: ndarray(int)
        simulation step list
    params: ndarray(float)
        cosmology parameters

    Returns
    -------
    ratio: ndarray(float)
        power spectrum ratio with respect to linear theory

    """
    z_vals = step_to_z(steps)
    lin_pk = []
    for z in z_vals:
        lin_pk.append(run_ccl_lin_pk(params, kvals, z))
    lin_pk = np.array(corr_vals)
    return pk_vals/lin_pk


def halofit_ratio(k_vals,pk_vals,steps,params):
    """
    ratio with respect to halofit

    Parameters
    ----------
    k_vals: ndarray(float)
        input wavenumbers
    pk_vals: ndarray(float)
        input power spectra
    steps: ndarray(int)
        simulation step list
    params: ndarray(float)
        cosmology parameters

    Returns
    -------
    ratio: ndarray(float)
        power spectrum ratio with respect to halofit model

    """
    z_vals = step_to_z(steps)
    halo_pk = []
    for z in z_vals:
        halo_pk.append(run_ccl_halofit_pk(params, kvals, z))
    halo_pk = np.array(corr_vals)
    return pk_vals/halo_pk

