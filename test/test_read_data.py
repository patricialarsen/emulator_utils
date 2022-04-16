"""
"""
import numpy as np

from emulator_utils.read_data import readpowerspec, readpolspice, readcorr

def test_readpowerspec():
    try:
        k,pk,err,nmodes = readpowerspec('test/data/powerspec_LJ/m000p.pk.499')
        assert True
    except:
        assert False 

def test_readpolspice():
    try:
        l, cl = readpolspice('test/data/cl/power_247_30_mask_105_120.cl')
        assert True
    except:
        assert False

def test_readcorr():
    try: 
        rmin, rmax, corr, count, binsum = readcorr('test/data/corr/m000-499.correlation_function.0')
        assert True
    except:
        assert False




