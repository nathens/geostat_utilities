# simulationFFT.py
# Created: April 6th, 2018
# Python implementation of GAIA lab's MGSimulFFT.m available here: http://wp.unil.ch/gaia/downloads/

import numpy as np
from numpy.fft import fftn, fftshift, ifftn
from numpy.random import uniform as rand

def simulFFT(x, y, z, mu, sigma2, m, lx , ly, lz):
    if z == 0: z = 1
    xx, yy, zz = np.meshgrid(np.arange(x), np.arange(y), np.arange(z))

    nx = xx.shape[1]
    ny = xx.shape[0]
    nz = xx.shape[2]

    xc = (xx[0, -1, 0] - xx[0, 0, 0]) / 2.0 # coordinates of the center of the grid
    yc = (yy[-1, 0, 0] - yy[0, 0, 0]) / 2.0
    zc = (zz[-1, 0, 0] - zz[0, 0, 0]) / 2.0

    lx = lx / 3.0
    ly = ly / 3.0
    lz = lz / 3.0

    h = np.sqrt(((xx - xc) / lx)**2 + ((yy - yc) / ly)**2 + ((zz - zc) / lz)**2)

    if m == 'Exponential':
        c = np.exp(-np.abs(h)) * sigma2
    elif m == 'Gaussian':
        c = np.exp(-(h**2)) * sigma2

    grid = fftn(fftshift(c)) / (nx*ny*nz)
    grid = np.abs(grid)
    grid[0, 0, 0] = 0
    ran = np.sqrt(grid) * np.exp(1j * np.angle(fftn(rand(size=(ny, nx, nz)))))
    grid = np.real(ifftn(ran*nx*ny*nz))
    std = np.std(grid)
    if x == 1 or y == 1 or z == 1: grid = np.squeeze(grid)
    return grid / std * np.sqrt(sigma2) + mu
