import pyccl

vc = 2.998e5 #(km/s), speed of light

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


