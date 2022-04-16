"""
"""
import numpy as np

from emulator_utils.power_class import PowerSpectrum

def test_initialize():
    try:
        power = PowerSpectrum()
        assert True
    except:
        assert False

def test_filelist():
    try:
        powerspec = PowerSpectrum()
        file_list = powerspec.set_file_list(direc='test/data/powerspec_LJ')
        print(file_list)
        print(len(file_list))
        assert len(file_list)==73
    except:
        assert False

def test_step():
    powerspec = PowerSpectrum()
    file_list = powerspec.set_file_list(direc='test/data/powerspec_LJ')
    steps = powerspec.set_steps()
    assert(len(steps)==73)

def test_data():
    powerspec = PowerSpectrum()
    powerspec.set_file_list(direc='test/data/powerspec_LJ')
    powerspec.set_steps()
    powerspec.set_data()
    assert (len(powerspec.k)==73)

def test_ratio():
    powerspec = PowerSpectrum()
    powerspec.set_file_list(direc='test/data/powerspec_LJ')
    powerspec.set_steps()
    powerspec.set_data()
    ratio = powerspec.set_conserved_quantities()
    assert (len(ratio)==73)

