#
# convenience functions for ccl 
#


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

import camb
from camb import model, initialpower
from camb.sources import GaussianSourceWindow, SplinedSourceWindow

vc = 2.998e5 #(km/s), speed of light

def run_camb_limber(params,win_func,chimax):
    nz = 1000 #number of steps to use for the radial/redshift integration
    kmax=10  #kmax to use

    omegaM = params[0]
    omegaB = params[1]
    sigma8 = params[2]
    h = params[3]
    n_s = params[4]
    omegaM = omegaM/h**2
    omegaB = omegaB/h**2
    omegaC = omegaM-omegaB

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=h*100., ombh2=omegaB*h**2, omch2=omegaC*h**2,TCMB=2.725)
    pars.InitPower.set_params(As=2e-9, ns=n_s, r=0)
    pars.set_for_lmax(2500, lens_potential_accuracy=4);
    pars.set_matter_power(redshifts=[0],kmax=kmax);
    results = camb.get_results(pars)
    s8_new = results.get_sigma8()
    fact = sigma8/s8_new
    pars.set_cosmology(H0=h*100., ombh2=omegaB*h**2, omch2=omegaC*h**2,TCMB=2.725)
    pars.InitPower.set_params(As=2e-9*fact**2, ns=n_s, r=0)
    pars.set_for_lmax(2500, lens_potential_accuracy=4);
    results = camb.get_background(pars)
    chistar = chimax
    #chistar = results.conformal_time(0)- results.tau_maxvis
    #chistar = 10000.
    chis = np.linspace(0,chistar,nz)
    zs=results.redshift_at_comoving_radial_distance(chis)
    #Calculate array of delta_chi, and drop first and last points where things go singular
    dchis = (chis[2:]-chis[:-2])/2
    chis = chis[1:-1]
    zs = zs[1:-1]

    PK = camb.get_matter_power_interpolator(pars, nonlinear=True,
        hubble_units=False, k_hunit=False, kmax=kmax,
        var1=model.Transfer_Weyl,var2=model.Transfer_Weyl, zmax=zs[-1])
    win = win_func(chis)
    #win = ((chistar-chis)/(chis**2*chistar))**2
    #Do integral over chi
    ls = np.arange(2,2500+1, dtype=np.float64)
    cl_kappa=np.zeros(ls.shape)
    w = np.ones(chis.shape) #this is just used to set to zero k values out of range of interpolation
    for i, l in enumerate(ls):
        k=(l+0.5)/chis
        w[:]=1
        w[k<1e-4]=0
        w[k>=kmax]=0
        cl_kappa[i] = np.dot(dchis, w*PK.P(zs, k, grid=False)*win/k**2)
    cl_kappa*= (ls*(ls+1))**2
    cl_limber= 4*cl_kappa/2/np.pi #convert kappa power to [l(l+1)]^2C_phi/2pi (what cl_camb is)
    return ls,cl_limber


def run_camb_lin_pk(params, kvals, z):
    '''
    Run the CAMB linear matter power spectrum on given set of parameters
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

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=h*100., ombh2=omegaB*h**2, omch2=omegaC*h**2,TCMB=2.725)
    pars.InitPower.set_params(As=2e-9, ns=n_s, r=0)
    pars.set_for_lmax(2500, lens_potential_accuracy=0);
    pars.set_matter_power(redshifts=[z],kmax=np.max(kvals));
    results = camb.get_results(pars)
    # resetting cosmology with sigma8 fixed rather than As
    s8_new = results.get_sigma8()
    fact = sigma8/s8_new
    pars.set_cosmology(H0=h*100., ombh2=omegaB*h**2, omch2=omegaC*h**2,TCMB=2.725)
    pars.InitPower.set_params(As=2e-9*fact**2, ns=n_s, r=0)
    pars.set_for_lmax(2500, lens_potential_accuracy=0);
    pars.set_matter_power(redshifts=[z],kmax=np.max(kvals));
    results = camb.get_results(pars)

    kh, z2, pk = results.get_matter_power_spectrum(minkh=np.min(kvals)/h, maxkh=np.max(kvals)/h, npoints = len(kvals))
    return pk[0]/h**3


def run_camb_halofit_pk(params, kvals, z):
    '''
    Run the CAMB linear matter power spectrum on given set of parameters
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

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=h*100., ombh2=omegaB*h**2, omch2=omegaC*h**2)
    pars.InitPower.set_params(As=2e-9, ns=n_s, r=0)
    pars.set_for_lmax(2500, lens_potential_accuracy=0);

    pars.set_matter_power(redshifts=[z],kmax=2.0);
    results = camb.get_results(pars)
    # resetting cosmology with sigma8 fixed rather than As
    s8_new = results.get_sigma8()
    fact = sigma8/s8_new
    pars.set_cosmology(H0=h*100., ombh2=omegaB*h**2, omch2=omegaC*h**2)
    pars.InitPower.set_params(As=2e-9*fact**2, ns=n_s, r=0)
    pars.set_for_lmax(2500, lens_potential_accuracy=0);

    pars.NonLinear = model.NonLinear_both
    pars.NonLinearModel.set_params(halofit_version='takahashi')
    pars.set_matter_power(redshifts=[z],kmax=np.max(kvals));
    results = camb.get_results(pars)

    kh, z2, pk = results.get_matter_power_spectrum(minkh=np.min(kvals)/h, maxkh=np.max(kvals)/h, npoints = len(kvals))
    return pk[0]/h**3

def alt_fn_camb(params, input_z, path_z,kmax,lmax):
    '''
    Run the CAMB weak lensing power spectrum as an alernate theory code
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
    if input_z:
        print('not implemented yet')
        z_n = np.linspace(0., 2., 500)
        n = pdz_arbitrary(path_z)
    else:
        zm = params[5]
        fwhm = params[6]
        #z_n = np.linspace(0., 2., 500)
        #n = pdz_given(z_n,zm,fwhm)

    pars = camb.CAMBparams()
    #pars = camb.set_params(limber_phi_lmin = 0)
    pars.set_cosmology(H0=h*100., ombh2=omegaB*h**2, omch2=omegaC*h**2)
    pars.InitPower.set_params(As=2e-9, ns=n_s, r=0)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0);
    #pars.set_params(limber_phi_lmin = 0)

    pars.set_matter_power(redshifts=[0],kmax=kmax);
    results = camb.get_results(pars)
    # resetting cosmology with sigma8 fixed rather than As
    s8_new = results.get_sigma8()
    fact = sigma8/s8_new
    pars.set_cosmology(H0=h*100., ombh2=omegaB*h**2, omch2=omegaC*h**2)
    pars.InitPower.set_params(As=2e-9*fact**2, ns=n_s, r=0)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1); # check this accuracy
    pars.Want_CMB = False
    pars.NonLinear = model.NonLinear_both
    pars.NonLinearModel.set_params(halofit_version='takahashi')
    pars.set_matter_power(redshifts=[0],kmax=kmax);


    pars.SourceWindows = [
    GaussianSourceWindow(redshift=zm, source_type='lensing', sigma=fwhm/2.)]
    #print(pars.SourceTerms)
    #pars.SourceTerms.limber_windows = True
    #pars.SourceTerms.limber_phi_lmin = 100  
    #print(pars.SourceTerms)


    results = camb.get_results(pars)
    ell=  np.arange(2, lmax+1)
    cls = results.get_source_cls_dict()['W1xW1'][2:lmax+1]

    return ell,cls


def run_camb_cls(params, ell, model_nlin='takahashi', path_z=''):
    '''
    Run the CAMB weak lensing power spectrum
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

    lmax = np.max(ell)
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=h*100., ombh2=omegaB*h**2, omch2=omegaC*h**2)
    pars.InitPower.set_params(As=2e-9, ns=n_s, r=0)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0);

    pars.set_matter_power(redshifts=[0],kmax=2.0);
    results = camb.get_results(pars)
    # resetting cosmology with sigma8 fixed rather than As
    s8_new = results.get_sigma8()
    fact = sigma8/s8_new
    pars.set_cosmology(H0=h*100., ombh2=omegaB*h**2, omch2=omegaC*h**2)
    pars.InitPower.set_params(As=2e-9*fact**2, ns=n_s, r=0)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1); # check this accuracy
    pars.Want_CMB = False
    pars.NonLinear = model.NonLinear_both
    #pars.NonLinearModel.set_params(halofit_version=model_nlin)
    pars.set_matter_power(redshifts=[0],kmax=10.);


    pars.SourceWindows = [
    GaussianSourceWindow(redshift=zm, source_type='lensing', sigma=fwhm/2.)]
    print(pars.SourceTerms)
    results = camb.get_results(pars)
    cls = results.get_source_cls_dict()['W1xW1'][2:lmax+1]

    return cls


'''
 - sigma definition of window:
        winamp =  exp(-((a-Win%a)/Win%sigma)**2/2)
        Window_f_a = a*winamp/Win%sigma/root2pi
   e- ((z -zm)/fwhm)**2
   fwhm = sigma*2 
    return 1./(np.sqrt(2*np.pi) * (fwhm/2.0)) * np.exp(-2.0*(z-zm)**2/fwhm**2)
'''
