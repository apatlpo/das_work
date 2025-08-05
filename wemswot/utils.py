import os
from glob import glob

import xarray as xr
import pandas as pd
import numpy as np


root_dir = "/Users/aponte/Current_Projects/das/data/wemswot"
diag_dir = os.path.join(root_dir, "diags")
current_dir = os.path.join(root_dir, "currents")
ctd_dir = os.path.join(root_dir, "ctds")


station_info = {
    1: dict(lat=43.062667, lon=5.890667, depth=41),
    2: dict(lat=43.056017, lon=5.891633, depth=68),
    3: dict(lat=43.027883, lon=5.896000, depth=118),
    4: dict(lat=43.005717, lon=5.897667, depth=906),
}

# ------------------------------ current meter data -------------------------------------------------

def read_aqd(aqp_dir, core_name, profile=True, first_cell=None, cell_size=None):

    # read sensor data first
    if profile:
        file = os.path.join(aqp_dir, f'{core_name}.sen')
    else:
        file = os.path.join(aqp_dir, f'{core_name}.dat')
    se = read_sensor_data(file, profile=profile)

    if profile:
        pr = xr.merge([read_profile_data(os.path.join(aqp_dir, f'{core_name}.{v}')).rename(v) for v in ["a1", "a2", "a3", "v1", "v2", "v3"]])
        pr["time"] = se.time
        pr = pr.assign_coords(
            time_days=(pr["time"]-pr["time"][0])/pd.Timedelta("1d"),
            z=("bin", first_cell+np.arange(pr.bin.size)*cell_size),
        )
        pr["z"].attrs.update(units="m")
        for v in ["v1", "v2", "v3"]:
            pr[v].attrs.update(units="m/s")
        pr = pr.transpose("bin", "time") # for plotting purposes
        
        pr["v1"].attrs.update(
            long_name="eastward velocity",
        )
        pr["v2"].attrs.update(
            long_name="northward velocity",
        )
        pr["v3"].attrs.update(
            long_name="upward velocity",
        )
        
        ds = xr.merge([pr, se])
        return ds
    else:
        return se

def read_sensor_data(file, profile=True):

    # 04 18 2023 05 59 00 00000000 00110000  13.4 1506.6 261.1  -1.7  -0.3  67.501  13.90     0     0
    time_vars = ["month", "day", "year", "hour", "minute", "second"]
    units = {
        "sound speed": "m/s",
        "heading": "degrees",
        "pitch": "degrees",
        "roll": "degrees",
        "pressure": "dbar",
        "temperature": "degrees C",
    }
    if profile:
        names = time_vars+[
            "error code", "status code", 
            "battery voltage", "sound speed", 
            "heading", "pitch", "roll", "pressure", "temperature",
            "analog input 1", "analog input 2", 
        ]
    else:
        names = time_vars+[
            "error code", "status code",
            "v1", "v2", "v3", "a1", "a2", "a3", 
            "battery voltage", "sound speed", "sound speed used",
            "heading", "pitch", "roll", "pressure", "pressure meters", "temperature",
            "analog input 1", "analog input 2",
            "speed", "direction",
        ]
        units.update(
            **{
                "v1": "m/s",
                "v2": "m/s",
                "v3": "mr/s",
                "sound speed used": "m/s",
                "speed": "m/s", 
                "direction": "degrees",
            }
        )

    df = pd.read_csv(
        file, header=None, 
        delimiter=r"\s+",
        names=names,
    )
    df["time"] = pd.to_datetime(df[time_vars])
    df = df.set_index("time")
    df = df.drop(columns=time_vars)
    se = df.to_xarray()
    
    for k, v in units.items():
        se[k].attrs.update(units=v)

    return se


def read_profile_data(file):
    df = pd.read_csv(file, header=None, delimiter=r"\s+").stack(future_stack=True)
    da = (
        df.to_xarray()
        .rename(level_0="time", level_1="bin")
    )
    return da



def compute_vector_principal_axes(u, v):
    """Compute vector time series principal axes
    See Emery and Thomson section 4.4.1

    Parameters
    ----------
        u, v: xr.DataArray
        Must possess `time` dimension

    Returns
    -------
        major, minor: xr.DataArray
        Complex arrays corresponding to major/minor axes scaled by respective eigenvalues
        The orientation of each axis is thus provided by the complex angle
    """

    # demean
    u_mean, v_mean = u.mean("time"), v.mean("time")
    up = u - u_mean
    vp = v - v_mean

    # build velocity covariance array and return eigenvectors
    uu = (up * up).mean("time")
    uv = (up * vp).mean("time")
    vv = (vp * vp).mean("time")

    # proceed with principal axes calculation  see Emery and Thomson 4.53
    ke = (uu + vv) * 0.5
    det = np.sqrt((uu - vv) ** 2 + 4 * uv**2)
    lambda_1 = ke + det * 0.5
    lambda_2 = ke - det * 0.5
    #
    s_1 = (lambda_1 - uu) / uv
    e_1 = (1 + 1j * s_1) / np.sqrt(1 + s_1**2)
    # compute vectors scaled by eigenvalues
    major = np.sqrt(lambda_1) * e_1
    minor = np.sqrt(lambda_2) * e_1 * np.exp(1j * np.pi / 2)

    return major, minor


def build_principal_ellipse(major, minor, scale=1):
    """build principal ellipse for drawing purposes"""
    # build ellipse
    t = xr.DataArray(np.linspace(0, 2 * np.pi, 100), dims="time").rename("time")
    v = (np.cos(t) * major + np.sin(t) * minor) * scale
    x = np.real(v)
    y = np.imag(v)
    return x, y