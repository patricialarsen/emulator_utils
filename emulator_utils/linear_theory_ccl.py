import os
import numpy as np
import subprocess as sp
from numba import jit
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM

from wlproject import base_numba
from wlproject.base_numba import source_redshift_dist as srd
from wlproject.base_numba import pade_funcs as pade
from scipy.integrate import fixed_quad
from scipy.interpolate import interp1d

from wlproject import run_pk_emu
from wlproject.run_model_cls import * 

import pyccl

vc = 2.998e5 #(km/s), speed of light


def get_pk_array_ccl(cosmo,ncosmo,zfunc,zmin,zmax,nz,nk,mink):
    pk_array = []
    lk_array = []
    cmin = ncosmo.comoving_distance(zmin).value
    cmax = ncosmo.comoving_distance(zmax).value
    c_array = np.linspace(cmin,cmax,nz+1)[1:]
    zl_array = zfunc(c_array)
    k_lo = np.logspace(np.log10(mink),10.0,nk)
    for zl_tmp in zl_array:
        pow_ccl = pyccl.nonlin_matter_power(cosmo,k_lo,1./(1.+zl_tmp))
        lk_array.append(k_lo)
        pk_array.append(pow_ccl)
    return zl_array,lk_array,pk_array


def get_ccl_cosmo(params,power):
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
    '''
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
    '''
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

def run_ccl_halofit_pk(params, kvals, z):
    '''
    Run the CCL nonlinear halofit matter power spectrum on given set of parameters
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
    nlinear: ndarray(float)
       P(k) values for nonlinear matter power spectrum
    '''
    omegaM = params[0]
    omegaB = params[1]
    sigma8 = params[2]
    h = params[3]
    n_s = params[4]
    omegaM = omegaM/h**2
    omegaB = omegaB/h**2
    omegaC = omegaM-omegaB
    cosmo = pyccl.Cosmology(transfer_function='boltzmann_camb', matter_power_spectrum='halofit', Omega_c = omegaC, Omega_b = omegaB, h = h, n_s = n_s, sigma8 = sigma8)
    nlinear = pyccl.nonlin_matter_power(cosmo,kvals,1./(1.+z))
    return nlinear


def run_ccl_emu_pk(params, kvals, z):
    '''
    Run the CCL nonlinear cosmic emu matter power spectrum on given set of parameters
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
    nlinear: ndarray(float)
       P(k) values for nonlinear matter power spectrum
    '''
    omegaM = params[0]
    omegaB = params[1]
    sigma8 = params[2]
    h = params[3]
    n_s = params[4]
    omegaM = omegaM/h**2
    omegaB = omegaB/h**2
    omegaC = omegaM-omegaB
    cosmo = pyccl.Cosmology(transfer_function='boltzmann_camb', matter_power_spectrum='emu', Omega_c = omegaC, Omega_b = omegaB, h = h, n_s = n_s, sigma8 = sigma8, Neff=3.04)
    nlinear = pyccl.nonlin_matter_power(cosmo,kvals,1./(1.+z))
    return nlinear

def alt_fn_ccl(params, input_z, path_z):
    '''
    Run the CCL weak lensing power spectrum as an alernate theory code
    Parameters
    ----------
    params: ndarray(float)
       parameters to compute power spectrum
    input_z: bool
       True if n(z) file input
    path_z: str
       path to n(z) file if applicable ('' if not)
    Returns
    -------
    ell: ndarray(int)
       ell values
    cls: ndarray(float)
       C(ell) values
    '''
    omegaM = params[0]
    omegaB = params[1]
    sigma8 = params[2]
    h = params[3]
    n_s = params[4]
    omegaM = omegaM/h**2
    omegaB = omegaB/h**2
    omegaC = omegaM-omegaB

    # too hard coded
    ell = np.arange(2,10000)
    if input_z:
        z_n = np.linspace(0., 2., 500)
        n = pdz_arbitrary(path_z)
    else:
        zm = params[5]
        fwhm = params[6]
        z_n = np.linspace(0., 2., 500)
        n = pdz_given(z_n,zm,fwhm)

    cosmo = pyccl.Cosmology(transfer_function='boltzmann_camb', matter_power_spectrum='emu', Omega_c = omegaC, Omega_b = omegaB, h = h, n_s = n_s, sigma8 = sigma8, Neff=3.04)

    lens1 = pyccl.WeakLensingTracer(cosmo, dndz=(z_n, n))
    cls = pyccl.angular_cl(cosmo, lens1, lens1, ell)

    return ell,cls

def run_ccl_cls(params, ell, model, path_z):
    '''
    Run the CCL weak lensing power spectrum
    Parameters
    ----------
    params: ndarray(float)
       parameters to compute power spectrum
    ell: ndarray(int)
       ell values 
    model: str
       model for nonlinear power spectrum: e.g. emu, halofit
    path_z: str
       path to n(z) file if applicable ('' if not)
    Returns
    -------
    cls: ndarray(float)
       C(ell) values
    '''
    omegaM = params[0]
    omegaB = params[1]
    sigma8 = params[2]
    h = params[3]
    n_s = params[4]
    omegaM = omegaM/h**2
    omegaB = omegaB/h**2
    omegaC = omegaM-omegaB


    if path_z=='':
        zm = params[5]
        fwhm = params[6]
        z_n = np.linspace(0., 2., 500)
        n = pdz_given(z_n,zm,fwhm)
    else:
        z_n = np.linspace(0., 2., 500)
        n = pdz_arbitrary(path_z)

    cosmo = pyccl.Cosmology(transfer_function='boltzmann_camb', matter_power_spectrum=model, Omega_c = omegaC, Omega_b = omegaB, h = h, n_s = n_s, sigma8 = sigma8, Neff=3.04)

    lens1 = pyccl.WeakLensingTracer(cosmo, dndz=(z_n, n))
    cls = pyccl.angular_cl(cosmo, lens1, lens1, ell)

    return cls

