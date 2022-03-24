from glob import glob

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import gsw


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

    #print(f, ds)

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

def plot_argo_profiles(ds):
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
        da.plot.line(y=z_coord, add_legend=False, ax=ax)
        ax.grid()
        ax.set_xlabel("")
        ax.set_title(v)
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
