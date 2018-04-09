# simulation.py
# Created: April 6th, 2018
# Python implementation of unconditional LU simulation

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import cholesky

def gaussian_variogram(h, sill, nugget, length):
    gamma = (sill - nugget) * (1 - np.exp(-3 * (h / length)**2)) + nugget
    gamma[h < length * 1e-8] = 0
    return gamma

def exponential_variogram(h, sill, nugget, length):
    gamma = (sill - nugget) * (1 - np.exp(-3 * h / length)) + nugget
    gamma[h < length * 1e-8] = 0
    return gamma

def spherical_variogram(h, sill, nugget, length):
    # ToDo: add nugget
    gamma = (sill - nugget) * (1.5 * h / length - 0.5 * (h / length)**3)
    gamma[h > length] = 1
    gamma[h < length * 1e-8] = 0
    return gamma

def calculate_variogram(h, mu, sill, nugget, length, model):
    if nugget == 0: nugget = 1e-8
    if model == 'Gaussian':
        return gaussian_variogram(h, sill, nugget, length)
    elif model == 'Exponential':
        return exponential_variogram(h, sill, nugget, length)
    elif model == 'Spherical':
        return spherical_variogram(h, sill, nugget, length)
    else:
        raise ValueError('Variogram model must be Gaussian, Exponential, or Spherical')

def unconditional_simulation(nx, ny, mu, sill, nugget, length, model):
    """ Performs unconditional simulation with specified mean, variance,
    and correlation length. 2D only.
    """
    xx = np.tile(np.arange(nx), ny)
    yy = np.repeat(np.arange(ny), nx)
    grid = np.stack((xx.ravel(), yy.ravel())).T
    h = squareform(pdist(grid)).flatten()
    variogram = calculate_variogram(h, mu, sill, nugget, length, model)
    cov_size = nx * ny
    covariance = (sill - variogram).reshape((cov_size, cov_size))
    low = cholesky(covariance).T
    u = np.random.normal(0, 1, size = cov_size)
    return (mu + np.matmul(low, u)).reshape((ny, nx))
