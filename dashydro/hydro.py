import os
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import gsw

# ------------------------------- paths ------------------------------------------

if os.getlogin()=="aponte":
    #root_data_dir = "/Users/aponte/Cloud/Dropbox/Public/das"
    root_data_dir = "/Users/aponte/Current_Projects/das/data"
elif os.getlogin()=="toto":
    root_data_dir = "/path/to/datadir"
else:
    print("You need to update root_data_dir in das_work/dashydo/hydro.py")

def raw_dir(deployment):
    """return path to deployment raw data dir"""
    return os.path.join(root_data_dir, deployment, "raw/")

def processed_dir(deployment):
    """return path to deployment processed data dir"""
    return os.path.join(root_data_dir, deployment, "processed/")

# ------------------------------- misc ------------------------------------------

toulon = dict(lon=5.9, lat=43.)

def get_f(lat):
    """ outputs the Coriolis frequency

    Parameters
    ----------
    lat: float
        Latitude where to compute the Coriolis frequency

    """
    f = 2 *2*np.pi/86164.1 * np.sin(lat*np.pi/180)
    f_cpd = f*86400/2/np.pi
    return f, f_cpd

# ------------------------------- WOA ------------------------------------------

woa_dir = os.path.join(root_data_dir, "woa")

def _load_woa(v, **kwargs):
    ds = xr.open_dataset(os.path.join(woa_dir, f"woa18_A5B7_{v}00_01.nc"),
                         decode_times=False,
                        )
    ds = ds.sel(**kwargs, method="nearest").squeeze()
    return ds


def load_woa(N2=True, **kwargs):
    """ load World Ocean Atlas hydrology data
    For details about the World Ocean Atlas (WOA) climatology, see:
        https://www.nodc.noaa.gov/OC5/woa18/

    Parameters
    ----------
    **kwargs:
        passed to ds.sel(...) for geographical selection (lon, lat, depth)
    N2: boolean, optional
        compute the square buoyancy frequency (default is True)
    """

    ds = xr.merge([_load_woa(v, **kwargs) for v in ["t", "s"]])
    ds = ds[["t_an", "s_an"]].rename(t_an="temperature", s_an="salinity")
    ds = ds.assign_coords(z=-ds.depth, p=gsw.p_from_z(-ds.depth, ds.lat))
    ds["SA"] = gsw.SA_from_SP(ds.salinity, ds.p, ds.lon, ds.lat)
    ds["CT"] = gsw.CT_from_t(ds.SA, ds.temperature, ds.p)
    ds["PT"] = gsw.pt_from_CT(ds.SA, ds.CT)
    ds["sigma0"] = gsw.sigma0(ds.SA, ds.CT)
    # derive N2
    if N2:
        N2, p_mid = gsw.Nsquared(ds.SA, ds.CT, ds.SA*0.+ds.p, lat=ds.SA*0.+ds.lat, axis=0)
        ds["p_mid"] = (("depth_mid",)+ds.SA.dims[1:], p_mid)
        ds["z_mid"] = (("depth_mid",), -(ds.depth.values[:-1]+ds.depth.values[1:])*.5)
        ds = ds.set_coords(["p_mid", "z_mid"])
        ds["N2"] = (("depth_mid",)+ds.SA.dims[1:], N2)

    return ds

# ------------------------------- emso -----------------------------------------

def load_emso_nc(file, resample=None):

    ds = xr.open_dataset(file)
    ds = (ds
          .assign_coords(row=ds.time)
          .drop("time")
          .rename(row="time")
          .sortby("time")
         )

    dt = (ds.time.diff("time")/pd.Timedelta("1m")).median() # in minutes
    print(f"Median sampling interval = {float(dt)} minutes")

    # should despike prior to resampling (in principle) ...
    # resample
    if resample:
        ds = ds.resample(time=resample).mean()

    return ds

# ------------------------------- argo -----------------------------------------

def load_argo_nc(f):
    """ load argo file and massage
    """
    ds = xr.open_dataset(f)

    # required for profile plots (weird ...)
    ds = ds.transpose("DEPTH", "TIME", "LATITUDE", "LONGITUDE", "POSITION")
    #
    ds = ds.assign_coords(z = -ds.DEPTH )
    ds["lon"] = ("TIME", ds.LONGITUDE.values)
    ds["lat"] = ("TIME", ds.LATITUDE.values)
    lon = ds.PRES*0 + ds.lon
    lat = ds.PRES*0 + ds.lat
    #
    ds["SA"] = gsw.SA_from_SP(ds.PSAL, ds.PRES, lon, lat) # absolute Salinity
    ds["CT"] = gsw.CT_from_t(ds.SA, ds.TEMP, ds.PRES) # conservative temperature
    ds["PT"] = gsw.pt_from_CT(ds.SA, ds.CT) # potential temperature
    ds["sigma0"] = gsw.sigma0(ds.SA, ds.CT)
    #
    N2, p_mid = gsw.Nsquared(ds.SA, ds.CT, ds.PRES, lat=0, axis=0)
    #print(N2.shape, N2[:,1], p_mid[:,2])
    #z_mid = gsw.z_from_p(p_mid, ds.lat.values)
    #ds["DEPTH_MID"] = (("DEPTH_MID"), -np.nanmean(z_mid, axis=1))
    ds["DEPTH_MID"] = (("DEPTH_MID"), ds.DEPTH.values[:-1])
    ds = ds.assign_coords(z_mid=-ds.DEPTH_MID)
    ds["N2"] = (("DEPTH_MID", "TIME"), N2)

    # filter bad profiles (custom)
    flag = ds.PSAL.fillna(0).mean("DEPTH")
    ds = ds.where(flag>0, drop=True)

    return ds

def plot_argo_profiles(ds, woa=None, ylim=None):
    """ plot basic variables
    """

    fig, axes = plt.subplots(2,2, figsize=(10,10))
    _axes = axes.flatten()

    variables = ["PT", "SA", "sigma0", "N2"]
    for v, ax in zip(variables, _axes[:len(variables)]):
        da = ds[v]
        if "z" in da.coords:
            z_coord = "z"
        else:
            z_coord = "z_mid"
        if woa is not None:
            woa[v].plot.line(y=z_coord, color="k", lw=2, add_legend=False, ax=ax, label="woa")
        da.plot.line(y=z_coord, add_legend=False, ax=ax)
        if ylim is None:
            ax.set_ylim(float(da[z_coord].min())-50,
                        float(da[z_coord].max())+10,
                        )
        ax.grid()
        ax.set_xlabel("")
        ax.set_title(v)
        ax.legend()
    return axes

def smooth(ds, dz=50, depth_max=1000):
    """ smooth vertical profile """
    depth_bins = np.arange(0, depth_max, dz)
    depth = (depth_bins[:-1] + depth_bins[1:])*0.5
    _ds = (ds
          .groupby_bins("DEPTH", bins=depth_bins, labels=depth)
          .mean()
          .rename(DEPTH_bins="DEPTH")
          .drop(["N2", "z_mid", "DEPTH_MID"])
         )
    _ds = _ds.assign_coords(z=-_ds.DEPTH)
    #
    N2 = (ds["N2"]
          .groupby_bins("DEPTH_MID", bins=depth_bins, labels=depth)
          .mean()
          .rename(DEPTH_MID_bins="DEPTH_MID")
         )
    N2 = N2.assign_coords(z_mid=-N2.DEPTH_MID)
    #
    ds = xr.merge([N2, _ds])
    return ds
