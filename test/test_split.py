"""
"""
import numpy as np
from emulator_utils.split import random_holdout


test_x = np.arange(0, 1, 0.01).reshape(-1, 2) + 100
test_y = np.sin(test_x)

def test_minmax():
    Xtrain, Xtest, ytrain, ytest = random_holdout(test_x, test_y, 0.2)
    assert np.all(np.isfinite(Xtrain))
    assert np.all(np.isfinite(Xtest))
    assert np.all(np.isfinite(ytrain))
    assert np.all(np.isfinite(ytest))
