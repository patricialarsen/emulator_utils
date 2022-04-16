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


