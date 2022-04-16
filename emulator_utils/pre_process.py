"""
pre_process.py
=============
Rescaling the inputs and outputs

"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline


__all__ = ("minmax", "standard", "standard_minmax", "log_standard", "unscale", "custom", )


def minmax(data1d_batch):
    """
    Transform features by scaling each feature to a given range. This estimator scales and translates each feature individually 
    such that it is in the given range on the training set, e.g. between zero and one.

    The transformation is given by::
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min
    
    where min, max = feature_range.
    
    This transformation is often used as an alternative to zero mean,
    unit variance scaling.

    Parameters
    ----------
    data1d_batch: float
        data to scale

    Returns
    -------
    scaled_data: float
        scaled data
    scaler: func
        scaling function

    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data1d_batch)
    
    return scaled_data, scaler



def standard(data1d_batch):
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    The standard score of a sample `x` is calculated as:
        z = (x - u) / s
    where `u` is the mean of the training samples or zero if `with_mean=False`,
    and `s` is the standard deviation of the training samples or one if `with_std=False`.
    
    Centering and scaling happen independently on each feature by computing
    the relevant statistics on the samples in the training set. 
    
    Standardization of a dataset is a common requirement for many
    machine learning estimators: they might behave badly if the
    individual features do not more or less look like standard normally
    distributed data (e.g. Gaussian with 0 mean and unit variance).

    Parameters
    ----------
    data1d_batch: float
        add explanation

    Returns
    -------
    scaled_data: float
        scaled data
    scaler: func
        scaling function

    """

    scaler = StandardScaler(with_mean=False)
    scaled_data = scaler.fit_transform(data1d_batch)

    return scaled_data, scaler


def standard_minmax(data1d_batch):
    """
    Apply Standardization first, then a min-max scaling between 0 and 1.

    Parameters
    ----------
    data1d_batch: float
        add explanation

    Returns
    -------
    scaled_data: float
        scaled data
    scaler: func
        scaling function

    """
    scaler = Pipeline([
        ('stdscaler', StandardScaler()), 
        ('minmax', MinMaxScaler(feature_range=(0, 1)))
        ])
    scaled_data = scaler.fit_transform(data1d_batch)

    return scaled_data, scaler


def _log_transform(data):
    """
    log transform 

    Parameters
    ----------
    data: float
        input data 

    Returns
    -------
    log10data: float
        output data

    """
    return np.log10(data)

def _inv_log_transform(data):
    """
    inverse log transform 

    Parameters
    ----------
    data: float 
        input data

    Returns
    -------
    10data: float
        output data

    """
    return 10**(data)


def log_standard(data1d_batch):
    """
    Apply log10 first, and then standardize the data.

    Parameters
    ----------
    data1d_batch: float
        add explanation

    Returns
    -------
    scaled_data: float
        scaled data
    scaler: func
        scaling function

    """
    transformer = FunctionTransformer(func = _log_transform, inverse_func = _inv_log_transform, validate=True, check_inverse = True)

    #scaled_data = transormfer.fit_transform(data1d_batch) # for log-only. 

    scaler = Pipeline([
        ('log', transformer),
        ('stdscaler', StandardScaler())
        ])

    scaled_data = scaler.fit_transform(data1d_batch)
    return scaled_data, scaler


def custom(data1d_batch, function, inverse_function):
    """
    Follow the steps in log_standard transformer. Not currently implemented

    """
    return NotImplemented

def unscale(scaled_data, scaler):
    """
    Takes processed data to the original raw format

    Parameters
    -------
    scaled_data: float
        scaled data
    scaler: func
        scaling function

    Returns
    ----------
    data1d_batch: float
        add explanation


    """
    unscaled_data = scaler.inverse_transform(scaled_data)
    return unscaled_data



def _scale01(data1d_batch):
    """
    Returns a standardized rescaling of the input values.
     by mean and variance of the training scheme

    Parameters
    ----------
    f: ndarray of shape (n, )

    Returns
    -------
    f_rescaled: ndarray of shape (5, )
        Rescaled Cosmological parameters

    """
    batch_mean = np.mean(data1d_batch, axis = 0)
    batch_std = np.std(data1d_batch, axis = 0)
    return (data1d_batch - batch_mean)/batch_std, batch_mean, batch_std


