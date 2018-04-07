# simulation.py
# Created: April 6th, 2018
# Python implementation of unconditional simulation
# Original code from Lijing Wang - Stanford University

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import cholesky

def gaussian_variogram(h, sill, vrange, nugget):
    s = sill - nugget
    r = vrange / np.sqrt(-np.log(0.1))
    gamma = nugget + s * (1 - np.exp(-(h / r)**2))
    gamma[h < vrange * 1e-8] = 0
    return gamma

def exponential_variogram(h, sill, vrange, nugget):
    s = sill - nugget
    r = -vrange / np.log(0.1)
    gamma = nugget + s * (1 - np.exp(-h / r))
    gamma[h < vrange * 1e-8] = 0
    return gamma

def spherical_variogram(h, sill, vrange, nugget):
    s = sill - nugget
    gamma = s * (1.5 * h / vrange - 0.5 * (h / vrange)**3)
    gamma[h > vrange] = 1
    gamma[h < vrange * 1e-8] = 0
    return gamma

def calculate_variogram(h, params):
    sill = params[0]
    vrange = params[1]
    nugget = params[3]
    if nugget == 0: nugget = 1e-8
    if params[4] == 'Gaussian':
        return gaussian_variogram(h, sill, vrange, nugget)
    elif params[4] == 'Exponential':
        return exponential_variogram(h, sill, vrange, nugget)
    elif params[4] == 'Spherical':
        return spherical_variogram(h, sill, vrange, nugget)
    else:
        raise ValueError('Variogram model must be Gaussian, Exponential, or Spherical')

def unconditional_simulation(grid_size, params):
    # grid_size = [50, 50]
    # params = [sill, range, mean, nugget, model]
    # returns simulation grid
    x = np.tile(np.arange(grid_size[0]), grid_size[1])
    y = np.repeat(np.arange(grid_size[1]), grid_size[0])
    grid = np.stack((x.ravel(), y.ravel())).T
    dist = squareform(pdist(grid)).flatten()
    variogram = calculate_variogram(dist, params)
    cov_size = grid_size[0] * grid_size[1]
    covariance = (params[0] - variogram).reshape((cov_size, cov_size))
    low = cholesky(covariance)
    u = np.random.normal(0, 1, size = cov_size)
    return params[2] + np.matmul(low.T,u)
