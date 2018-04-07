# simulationFFT.py
# Created: April 6th, 2018
# Python implementation of GAIA lab's MGSimulFFT.m available here: http://wp.unil.ch/gaia/downloads/

import numpy as np
from numpy.fft import fftn, fftshift, ifftn
from numpy.random import uniform as rand
import sys

def simulFFT(nx, ny, nz, mu, sill, m, lx , ly, lz):
    if nz == 0: nz = 1 # 2D case
    xx, yy, zz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz))
    points = np.stack((xx.ravel(), yy.ravel(), zz.ravel())).T
    centroid = points.mean(axis = 0)
    length = np.array([lx, ly, lz]) / 3.0
    h = np.linalg.norm((points - centroid) / length, axis=1).reshape((ny, nx, nz))

    if m == 'Exponential':
        c = np.exp(-np.abs(h)) * sill
    elif m == 'Gaussian':
        c = np.exp(-(h**2)) * sill

    grid = fftn(fftshift(c)) / (nx*ny*nz)
    grid = np.abs(grid)
    grid[0, 0, 0] = 0 # reference level
    ran = np.sqrt(grid) * np.exp(1j * np.angle(fftn(rand(size=(ny, nx, nz)))))
    grid = np.real(ifftn(ran * nx * ny * nz))
    std = np.std(grid)
    if nx == 1 or ny == 1 or nz == 1: grid = np.squeeze(grid)
    return grid / std * np.sqrt(sill) + mu
