"""
Loading pretrained GPflow, sklearn, tensorflow models
"""

import numpy as np
import pickle
import os
from sklearn.decomposition import PCA
import gpflow


DEFAULT_PCA_RANK = 6

__all__ = ("gp_model_load", "tf_model_load", )

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))


def gp_model_load(filename):
    """Returns pretrained GP and PCA models (saved in ./models/) for given snapshot.

    Parameters
    ----------

    snap_ID: int between 0 to 99
        Corresponds to different time stamps

    nRankMax: int
        Number of truncated PCA bases. Only valid nRankMax for now is 6.

    Returns
    _______

    GPm: GPflow predictor object
    PCAm: sklearn predictor object.

    """

    ctx_for_loading = gpflow.saver.SaverContext(autocompile=False)
    saver = gpflow.saver.Saver()
    GPm = saver.load(filename, context=ctx_for_loading)
    GPm.clear()
    GPm.compile()
    return GPm



def sklearn_model_load(filename):
    '''
    Only PCA available right now

    Parameters

    Pickle file with trained PCA

    Returns
    sklearn model object

    '''

    PCAm = pickle.load(open(filename, 'rb'))

    return PCAm


def tf_model_load(filename):
    return NotImplemented


