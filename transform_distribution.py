# transform_distribution.py
# Author: Noah Athens
# Created: April 13th, 2018


def transform_distribution(grid, new_distribution):
    """ Transforms grid to new distribution."""
    old_distribution = np.sort(grid.flatten())
    new_distribution = np.sort(np.random.choice(new_distribution, size = grid.size))
    d = dict(zip(old_distribution, new_distribution))
    new_grid = np.vectorize(d.get)(grid)
    return new_grid
