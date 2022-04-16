"""
running emulator pk for validation

"""

import numpy as np
import subprocess as sp
import os
import os.path

def create_pk_emu(emu,j):

    os.system('rm '+emu.output_dir +'/pk_outputs/*');
    
    dzl = (emu.z_max-emu.z_min)/emu.num_z
    zl_array = np.linspace(emu.z_min,emu.z_max,emu.num_z)
    params = emu.params

    omegaM = params[j][0]
    omegaB = params[j][1]
    sigma8 = params[j][2]
    h = params[j][3]
    n_s = params[j][4]

    pk_outfile = emu.output_dir+'/pk_outputs'

    for i in range(emu.num_z):
        filename = pk_outfile+'/'+str(i)+"_"+str(zl_array[i])+"_"+str(dzl)+"_cosmo_"+str(j)+"_pk.dat"
        cmd = emu.emu_dir + "/emu.exe "+filename+" "+str(omegaB)+" "+str(omegaM)+" "+str(n_s)+" "+str(h*100)+" -1.0 "+str(sigma8)+" "+str(zl_array[i])
        if os.path.exists(emu.emu_dir+'/emu.exe'):
            sp.call(cmd,shell=True)
        else:
            raise Exception("emu.exe doesn't exist, have you run the emulator makefile?")
    return 0


def create_pk(pk_outfile,params,j,num_params,zmin,zmax,nbins,emu_dir):
    '''
    Create franken-emu outputs for a given cosmology and redshift range
    Params
    ------
    pk_outfile : str
        Path for creation of power spectra
    params : ndarray
        Parameter array for power spectra creation
    j : int
        index for current output 
    num_params : int 
        number of parameters 
    zmin : float
        minimum redshift
    zmax : float 
        maximum redshift
    nbins : int
        number of redshift bins
    emu_dir : str
        path to emulator directory 
    Returns 
    -------
    err_code : int
        Zero indicates proper execution
    Raises
    ----------
    Exception 
        If the emulator executable file doesn't exist 
    '''
    dzl = (zmax-zmin)/nbins
    zl_array = np.linspace(zmin,zmax,nbins)
    #zl_array = np.linspace(zmin,zmax-dzl,nbins)+dzl/2.0
    omegaM = params[j][0]
    omegaB = params[j][1]
    sigma8 = params[j][2]
    h = params[j][3]
    n_s = params[j][4]


    if num_params==7:
        z_m = params[j][5]
        fwhm = params[j][6]
    for i in range(nbins):
        filename = pk_outfile+'/'+str(i)+"_"+str(zl_array[i])+"_"+str(dzl)+"_cosmo_"+str(j)+"_pk.dat"
        cmd = emu_dir + "/emu.exe "+filename+" "+str(omegaB)+" "+str(omegaM)+" "+str(n_s)+" "+str(h*100)+" -1.0 "+str(sigma8)+" "+str(zl_array[i])
        if os.path.exists(emu_dir+'/emu.exe'):
            sp.call(cmd,shell=True)
        else:
            raise Exception("emu.exe doesn't exist, have you run the emulator makefile?")
    return 0
