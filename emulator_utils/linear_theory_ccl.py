"""
linear_theory_ccl.py
====================
Linear theory routines from ccl

"""

import pyccl
from scipy.interpolate import interp1d
import numpy as np

def get_ccl_cosmo(params,power):
    """
    gets ccl cosmology object 

    Parameters
    ----------
    params: ndarray(ndarray(float))
        parameter inputs 
    power: str
        power spectrum model

    Returns
    -------
    cosmo: ccl cosmology object

    """
    omegaM = params[0]
    omegaB = params[1]
    sigma8 = params[2]
    h = params[3]
    n_s = params[4]
    omegaM = omegaM/h**2
    omegaB = omegaB/h**2
    omegaC = omegaM-omegaB

    cosmo = pyccl.Cosmology(transfer_function='boltzmann_camb', matter_power_spectrum=power, Omega_c = omegaC, Omega_b = omegaB, h = h, n_s = n_s, sigma8 = sigma8,Neff=3.04)
    return cosmo


def run_ccl_lin_pk(params, kvals, z):
    """
    Run the CCL linear matter power spectrum on given set of parameters

    Parameters
    ----------
    params: ndarray(float)
       parameters to compute power spectrum
    kvals: ndarray(float)
       k values (Mpc^-1)
    z: float
       redshift

    Returns
    -------
    linear: ndarray(float)
       P(k) values for linear matter power spectrum

    """
    omegaM = params[0]
    omegaB = params[1]
    sigma8 = params[2]
    h = params[3]
    n_s = params[4]
    omegaM = omegaM/h**2
    omegaB = omegaB/h**2
    omegaC = omegaM-omegaB
    cosmo = pyccl.Cosmology(transfer_function='boltzmann_camb', matter_power_spectrum='linear', Omega_c = omegaC, Omega_b = omegaB, h = h, n_s = n_s, sigma8 = sigma8)
    linear = pyccl.linear_matter_power(cosmo,kvals,1./(1.+z))
    return linear

def weighted_func(halo,emu,num,kvals):
    """
    weighting function 

    Parameters
    ----------
    halo: ndarray(float)
        first input to weight
    emu: ndarray(float)
        second input to weight
    num: int
        number input to scaling
    kvals: ndarray(float)
        wavenumber input

    Returns
    -------
    pow_t: ndarray(float)
        weighted power

    """
    weight1 = np.exp(-(kvals*num))
    weight2 = (1.-weight1)
    pow_t = halo*(weight2) + emu*(weight1)
    return pow_t



def linear_addition(k,pk,new_k,k_handover,params,z):
    """
    linear theory addition scaling linear theory to retain value at large scales
    fix this

    Parameters
    ----------
    halo: ndarray(float)
        first input to weight
    emu: ndarray(float)
        second input to weight
    num: int
        number input to scaling
    kvals: ndarray(float)
        wavenumber input

    Returns
    -------
    pow_t: ndarray(float)
        weighted power

    """
    linear = run_ccl_lin_pk(params, new_k, z)
    pk_interp = interp1d(np.log10(k),np.log10(pk),bounds_error=False,fill_value=0.0)
    pk_weighted = 10**(pk_interp(np.log10(new_k)))
    mask_k = (new_k<k_handover[1])&(new_k>k_handover[0])
    weight = np.mean(pk_weighted[mask_k])/np.mean(linear[mask_k])
    k_swap = (k_handover[1]+k_handover[0])/2.
    return linear*weight*(new_k<k_swap) + pk_weighted*(new_k>=k_swap)

def linear_addition_weighted(k,pk,new_k,k_handover,params,z):
    """
    linear theory addition using a weighted average of linear and measured solutions
    fix this

    Parameters
    ----------
    halo: ndarray(float)
        first input to weight
    emu: ndarray(float)
        second input to weight
    num: int
        number input to scaling
    kvals: ndarray(float)
        wavenumber input

    Returns
    -------
    pow_t: ndarray(float)
        weighted power

    """
    linear = run_ccl_lin_pk(params, new_k, z)
    pk_interp = interp1d(np.log10(k),np.log10(pk),bounds_error=False,fill_value=0.0)
    pk_weighted = 10**(pk_interp(np.log10(new_k)))
    combination = weighted_func(linear,pk_weighted,1./k_handover,new_k)
    return combination


