import numpy as np
import xarray as xr

def get_mu(omega, f, N):
    """ get internal wave group velocity steepness
    This is also equals to -k/m where k,m are horizontal wavenumbers respectively
    The output is positive but the sign may be reversed.

    Parameters
    ----------
    omega: float, array, ...
        wave frequency (rad/s)
    f: float, array, ...
        Coriolis frequency (rad/s)
    N: float, array, ...
        Buoyancy frequency (rad/s)
    """
    return np.sqrt( (omega**2 - f**2)/(N**2-omega**2)  )

def get_lambda(omega, f, N, gamma):
    """ Bottom reflection parameter
    See Gerkema section 6.2

    Parameters
    ----------
    omega: float, array, ...
        wave frequency (rad/s)
    f: float, array, ...
        Coriolis frequency (rad/s)
    N: float, array, ...
        Buoyancy frequency (rad/s)
    gamma: float, array, ...
        Bathymetric slope (seafloor is at z = gamma x)
    """
    mu = get_mu(omega, f, N)
    return (gamma-mu)/(gamma+mu)
