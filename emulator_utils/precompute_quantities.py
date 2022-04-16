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
    correlation function ratio (not currently working)

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
        power spectrum ratio with respect to the final redshift


    """
    nsteps = len(steps)
    assert([r_min[i]==r_min[j] for i,j in range(nsteps)].all())
    assert([r_max[i]==r_max[j] for i,j in range(nsteps)].all())
    base_step = np.max(steps)
    base_idx = np.where(steps==base_step)[0]
    return corr_vals/corr_vals[base_idx]


