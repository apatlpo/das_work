
import os
import csv
#from pyproj import Proj, transform
from pyproj import Transformer
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#import pylab as pl
from IPython import display

from cartopy import crs as ccrs, feature as cfeature
import cartopy.io.img_tiles as cimgt
crs = ccrs.PlateCarree()

from utide import solve, reconstruct


# paths to data files
#cable_file = "/......./Interpolation_bathy_Nathan.csv"
#bathy_file = "/....../MNT_MED100m_GDL-CA_HOMONIM_WGS84_NM_ZNEG.grd"
#das_file = "/....../DATA_HA_1000m.nc"      # <---- Choice : path DAS data
cable_file = "/Users/aponte/Current_Projects/das/data/AH_DAS_LF_GL_1000m/Interpolation_bathy_Nathan.csv"
bathy_file = "/Users/aponte/Current_Projects/das/data/AH_DAS_LF_GL_1000m/MNT_MED100m_GDL-CA_HOMONIM_WGS84_NM_ZNEG.grd"
das_file = "/Users/aponte/Current_Projects/das/data/AH_DAS_LF_GL_1000m/DATA_HA_1000m.nc"      # <---- Choice : path DAS data


# plotting parameters
request = cimgt.OSM()
# Parameters for Offshore section of LSPM cable
lon_inf = 5.85
lat_inf = 42.77
lon_sup = 6.04
lat_sup = 43.11
delta = 0.0012
# Bounds: (lon_min, lon_max, lat_min, lat_max):
extent = [lon_inf, lon_sup , lat_inf, lat_sup] 

def load_cable_position():
    """load cable position data"""

    # load data in xarray dataset
    ds = (
        pd.read_csv(cable_file)
        .set_index("distance")
        .to_xarray()
        .rename(angle="azimuth", Z="bathymetry")
    )

    # compute lon/lat
    transformer = Transformer.from_crs('epsg:2154', 'epsg:4326')
    _lat, _lon = transformer.transform(ds["X"],ds["Y"])
    #ds = ds.assign_coords(longitude=("distance", _lon), latitude=("distance", _lat))
    ds["longitude"] = ("distance", _lon)
    ds["latitude"] = ("distance", _lat)

    # select rang of channels based on distance
    #ds = ds.isel(channel=slice(0,None,218)) # not necessary
    ds = ds.where( (ds.distance>=0) & (ds.distance<=52500), drop=True)

    return ds

def load_bathy():
    """load bathymetry"""

    ds = (
        xr.open_dataset(bathy_file)
        .rename(z="bathymetry")
    )
    ds = ds.sel(lon=slice(lon_inf, lon_sup), lat=slice(lat_inf, lat_sup))
    ds = ds.rename(lon="longitude", lat="latitude")

    return ds


def solve_and_predict(z, time, lat, dev=False, **kwargs):
    """ solve for tidal harmonics and create a tidal prediction"""

    _kwargs = dict(lat=lat)
    _kwargs.update(kwargs)
    coef = solve(time, z, **_kwargs)
    tide = reconstruct(time, coef, verbose=False)

    if dev:
        return coef, tide

    ha = _coef_to_xr(coef)
    
    pr = xr.Dataset(
        dict(
            z_in=("time", z),
            z_out=("time", tide["h"]),
        ),
        coords=dict(time=time),
    )
    
    return xr.merge([ha, pr])

def _coef_to_xr(coef):
    omega = coef["aux"]["frq"] # cph
    ds = xr.Dataset(
        dict(
            A=("constituent", coef["A"]),
            A_ci=("constituent", coef["A_ci"]),
            g=("constituent", coef["g"]),
            g_ci=("constituent", coef["g_ci"]),
            SNR=("constituent", coef["SNR"]),
        ),
        coords=dict(
            constituent=coef["name"],
            omega_cph=("constituent", omega),
            omega_cpd=("constituent", omega*24),
            omega=("constituent", 2*np.pi*omega/3600),
            power_nergy = ("constituent", coef["PE"]),
        ),
    )
    return ds

def tidal_analysis(das, di=1, **kwargs):
    """ loops over channels to compute the harmonic response """

    D = []
    for d in das.distance[::di]:
        _das = das.sel(distance=d)
        _ti = solve_and_predict(_das.strain.data, _das.time.data, float(_das.latitude), **kwargs)
        D.append(_ti)
    
    # concatenate results
    ti = (
        xr.concat(D, dim="distance")
        .assign_coords(**das.isel(distance=slice(0,None,di)).coords)
    )

    return ti


def plot(ha, cst=None):
    """ plot amplitude as a function of distance """

    if cst is not None:
        ha = ha.sel(constituent=cst)
    else:
        cst = ha.constituent

    cste = 0
        
    fig, ax = plt.subplots(
        3, 2, figsize=(32,8), 
        subplot_kw={'projection': crs}, 
        gridspec_kw={'height_ratios': [1,0.8,0.5],"width_ratios":[0.8,1]}
    )    
    gs = ax[0,1].get_gridspec()
    ax[0,0].remove()
    ax[1,0].remove()
    ax[2,0].remove()
    ax[1,1].remove()
    ax[2,1].remove()

    # bathy
    ax0 = fig.add_subplot(gs[:,0], projection=crs)
    im = ba["bathymetry"].plot.contourf(
        x="longitude", y="latitude", 
        levels=[0,-100,-200,-500,-1000,-1500,-2000],
        transform=ccrs.PlateCarree(),
    )
    axins = inset_axes(ax0,
                    width="100%",  
                    height="5%",
                    loc='lower center',
                    borderpad=-6)
    
    cbar = plt.colorbar(im,cax=axins, orientation="horizontal",fraction=0.05, pad=0.01)
    cbar.set_label(r"Bathymetry [m]",fontsize=16)
    ax0.set_xlabel("Longitude",fontsize=16)
    ax0.xaxis.set_label_position('top') 
    ax0.set_ylabel("Latitude",fontsize=16)
    ax0.set_xticks(np.arange(lon_inf,lon_sup,0.03))
    ax0.set_yticks(np.arange(lat_inf,lat_sup,0.04))
    ax0.set_xlabel('Longitude [°]', fontsize=15)
    ax0.set_ylabel('Latitude [°]', fontsize=15)
    ax0.tick_params(labelsize=12)
    ax0.gridlines()

    ax0.plot(ha.longitude,ha.latitude,'k*',transform=crs)
    #ax0.plot(ha.longitude_cable.isel[cste],latitude_cable[cste],'r*',transform=ccrs.PlateCarree())
    ax0.legend(["LSPM cable","Channel position"],loc='best',fontsize=12)
    ax0.set_xlim((lon_inf,lon_sup))
    ax0.set_ylim((lat_inf,lat_sup))
    
    ax1 = fig.add_subplot(gs[:2,1])
    A_max = float(ha["A"].max())
    print(f"Max amplitude = {A_max:.2e}")
    for i, c in enumerate(cst):
        _ha = ha.sel(constituent=c)
        # mask points where amplitude is smaller than CI
        _ha["A"] = _ha["A"].where( _ha["A"]>_ha["A_ci"] )
        ax1.fill_between(_ha.distance, _ha.A-_ha.A_ci, _ha.A+_ha.A_ci, alpha=.5, label=c)
    ax1.set_ylabel("Amplitude",fontsize=16)
    ax1.set_yscale("log")
    ax1.set_ylim(1e-5, 1e-1)
    ax1.legend()
        
    ax3 = fig.add_subplot(gs[2,1])
    ax3.plot(10**(-3)*ha.distance, ha.bathymetry,'k-*', linewidth=3)
    ax3.set_xlabel("Distance [km]",fontsize=16)
    ax3.set_ylabel("Bathymetry [m]",fontsize=16)
    #ax3.plot(10**(-3)*distance_cable.iloc[cste],bathymetry_cable.iloc[cste],'r*', linewidth=3)
    ax3.tick_params(axis='both', labelsize=16)
    ax3.grid()
    plt.setp(ax3.spines.values(), linewidth=3)


if __name__ == "__main__":

    # load cable information & regional bathymetry
    ca = load_cable_position()
    ba = load_bathy()
    
    # load das and combine with interpolated cable position data
    das = (
        xr.open_dataarray(das_file)
        .rename("strain").to_dataset()
    )
    das = das.assign_coords(**ca.interp(distance=das.distance))

    # flag to turn on/off computation and storage
    overwrite = False
    
    # with all constituents
    nc = "das_tidal_analysis_full.nc"
    if not os.path.isfile(nc) or overwrite:
        ha = tidal_analysis(das)
        ha.to_netcdf(nc, mode="w")
    else:
        ha = xr.open_dataset(nc)
    
    # with 6 constituents
    cst_6 = ["M2", "S2", "N2", "K1","O1", "P1",]
    _nc = nc.replace("_full", "_6")
    if not os.path.isfile(_nc) or overwrite:
        ha6 = tidal_analysis(das, constit=cst_6)
        ha6.to_netcdf(_nc, mode="w")
    else:
        ha6 = xr.open_dataset(_nc)
    
    # with 9 constituents
    cst_9 = cst_6+["K2", "M4", "MS4"]
    _nc = nc.replace("_full", "_9")
    if not os.path.isfile(_nc) or overwrite:
        ha9 = tidal_analysis(das, constit=cst_9)
        ha9.to_netcdf(_nc, mode="w")
    else:
        ha9 = xr.open_dataset(_nc)

    # quick plot to inspect ... should print figure also
    plot(ha, cst=["M2", "S2", "N2"])

