import numpy as np


def normalize_to_unit_interval(x, minmax):
    """Normalize 1D data x into [0,1] given (xmin, xmax)."""
    xmin, xmax = minmax
    return (x - xmin) / (xmax - xmin)

def chi_square_1d(x01, m):
    """
    Chi-square uniformity test for 1D data normalized to [0,1].
    
    Args:
        x01 : array-like, shape (n,)
            1D points in [0,1].
        m : int
            Number of bins.

    Returns:
        stat : float
            Chi-square statistic.
        dof : int
            Degrees of freedom (m - 1).
    """
    assert np.all((x01 >= 0) & (x01 <= 1)), "Data points must be in [0,1] (Use normalize_to_unit_interval)"

    n = len(x01) 
    bins = np.linspace(0, 1, m + 1)
    counts, _ = np.histogram(x01, bins=bins)
    E = n / m
    stat = np.sum((counts - E)**2 / np.maximum(E, 1e-12))
    dof = m - 1

    return stat, dof
