"""
"""
import numpy as np
from emulator_utils.pre_process import minmax, standard, standard_minmax, log_standard, unscale, custom


test_data1d = np.sin(np.arange(0, 1, 0.01).reshape(-1, 2))
test_nonzero = np.sin(np.arange(0, 1, 0.01).reshape(-1, 2)) + 100

def test_minmax():
    scaled, _ = minmax(test_data1d)
    assert np.all(np.isfinite(scaled))

def test_standard():
    scaled, _ = standard(test_data1d)
    assert np.all(np.isfinite(scaled))

def test_standard_minmax():
    scaled, _ = standard_minmax(test_data1d)
    assert np.all(np.isfinite(scaled))
'''
def test_log_standard():
    scaled, _ = log_standard(test_data1d)
    assert np.all(np.isfinite(scaled))
'''

def test_unscale():
    scaled, scaler = log_standard(test_nonzero)
    unscaled = unscale(scaled, scaler)
    np.testing.assert_almost_equal(test_nonzero, unscaled, 3)
