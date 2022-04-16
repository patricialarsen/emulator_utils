"""
linear_theory_camb.py
=====================
Linear theory functions for camb

"""
#
# convenience functions for camb
#

import numpy as np
import camb

def run_camb_lin_pk(params, kvals, z):
    """
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

    """
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
