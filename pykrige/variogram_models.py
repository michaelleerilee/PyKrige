from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

__doc__ = """
PyKrige
=======

Code by Benjamin S. Murphy and the PyKrige Developers
bscott.murphy@gmail.com

Summary
-------
Function definitions for variogram models. In each function, m is a list of
defining parameters and d is an array of the distance values at which to
calculate the variogram model.

References
----------
.. [1] P.K. Kitanidis, Introduction to Geostatistcs: Applications in
    Hydrogeology, (Cambridge University Press, 1997) 272 p.

Copyright (c) 2015-2018, PyKrige Developers
"""

###

class variogram_model(object):
    def __init__(self\
                 ,function=None\
                 ,parameters=None\
                 ,parameter_names=None\
                 ,bnds=None\
                 ,x0=None\
                 ):
        self.name            = function.__name__
        self.function        = function
        self.parameters      = parameters
        self.parameter_names = parameter_names
        self.bnds            = bnds
        self.x0              = x0
        return
    
###

variogram_models={}

def _add_model(model):
    variogram_models[model.name]=model

###

def linear_variogram_model(m, d):
    """Linear model, m is [slope, nugget]"""
    slope = float(m[0])
    nugget = float(m[1])
    return slope * d + nugget

def linear_variogram_model_bnds(lags=None,semivariance=None):
    return ([0., 0.], [np.inf, np.amax(semivariance)])

def linear_variogram_model_x0(lags=None,semivariance=None):
    return [(np.amax(semivariance) - np.amin(semivariance)) /
              (np.amax(lags) - np.amin(lags)), np.amin(semivariance)]

_add_model(
    variogram_model(function         = linear_variogram_model
                    ,parameter_names = ['slope','nugget']
                    ,bnds            = linear_variogram_model_bnds
                    ,x0              = linear_variogram_model_x0))

def power_variogram_model(m, d):
    """Power model, m is [scale, exponent, nugget]"""
    scale = float(m[0])
    exponent = float(m[1])
    nugget = float(m[2])
    return scale * d**exponent + nugget

def power_variogram_model_bnds(lags=None,semivariance=None):
    return ([0., 0.0001, 0.], [np.inf, 1.999, np.amax(semivariance)])

def power_variogram_model_x0(lags=None,semivariance=None):
    return [(np.amax(semivariance) - np.amin(semivariance)) /
              (np.amax(lags) - np.amin(lags)), 1.1, np.amin(semivariance)]

_add_model(
    variogram_model(function         = power_variogram_model
                    ,parameter_names = ['scale','exponent','nugget']
                    ,bnds            = power_variogram_model_bnds
                    ,x0              = power_variogram_model_x0))

def gaussian_variogram_model(m, d):
    """Gaussian model, m is [psill, range, nugget]"""
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return psill * (1. - np.exp(-d**2./(range_*4./7.)**2.)) + nugget

def gaussian_variogram_model_bnds(lags=None,semivariance=None):
    return  ([0., 0., 0.], [10.*np.amax(semivariance), np.amax(lags),
                               np.amax(semivariance)])

def gaussian_variogram_model_x0(lags=None,semivariance=None):
    return [np.amax(semivariance) - np.amin(semivariance),
              0.25*np.amax(lags), np.amin(semivariance)]

_add_model(
    variogram_model(function         = gaussian_variogram_model
                    ,parameter_names = ['psill','range_','nugget']
                    ,bnds            = gaussian_variogram_model_bnds
                    ,x0              = gaussian_variogram_model_x0))
    
def exponential_variogram_model(m, d):
    """Exponential model, m is [psill, range, nugget]"""
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return psill * (1. - np.exp(-d/(range_/3.))) + nugget

_add_model(
    variogram_model(function=exponential_variogram_model
                    ,parameter_names=['psill','range_','nugget']
                    ,bnds            = gaussian_variogram_model_bnds
                    ,x0              = gaussian_variogram_model_x0))    

def spherical_variogram_model(m, d):
    """Spherical model, m is [psill, range, nugget]"""
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return np.piecewise(d, [d <= range_, d > range_],
                        [lambda x: psill * ((3.*x)/(2.*range_) - (x**3.)/(2.*range_**3.)) + nugget, psill + nugget])
_add_model(
    variogram_model(function=spherical_variogram_model
                    ,parameter_names=['psill','range_','nugget']
                    ,bnds            = gaussian_variogram_model_bnds
                    ,x0              = gaussian_variogram_model_x0))                    

def hole_effect_variogram_model(m, d):
    """Hole Effect model, m is [psill, range, nugget]"""
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return psill * (1. - (1.-d/(range_/3.)) * np.exp(-d/(range_/3.))) + nugget

_add_model(
    variogram_model(function=hole_effect_variogram_model
                    ,parameter_names=['psill','range_','nugget']
                    ,bnds            = gaussian_variogram_model_bnds
                    ,x0              = gaussian_variogram_model_x0))
                    

def gamma_rayleigh_nuggetless_variogram_model(m,d):
    sill    = np.float(m[0])
    falloff = np.float(m[1])
    beta    = np.float(m[2])
    fd      = falloff*d
    omfd    = 1.0-falloff*d
    bfd2    = beta*omfd*omfd
    return  sill*fd*np.exp(omfd-bfd2)

def gamma_rayleigh_nuggetless_variogram_model_bnds(lags=None,semivariance=None):
    return ([0.0,0.0,0.0], [1000,10,10])
    
def gamma_rayleigh_nuggetless_variogram_model_x0(lags=None,semivariance=None):
    return [2.0,0.01,0.0001]
    
_add_model(
    variogram_model(function          = gamma_rayleigh_nuggetless_variogram_model
                    ,parameter_names  = ['sill','falloff','beta']
                    ,bnds             = gamma_rayleigh_nuggetless_variogram_model_bnds
                    ,x0               = gamma_rayleigh_nuggetless_variogram_model_x0))
