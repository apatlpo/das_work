#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:39:45 2025

@author: mohammedi
"""

###############################################################################
#######  HARMONICD ANALYSIS OF TOULON LSPM CABLE LOW FREQUENCY DATA WITH  OPTODAS ASN DURING 2023 ######
###############################################################################

#%%
###############################################################################
######################### LOAD OF MODULES #####################################
###############################################################################

import os
import csv
from pyproj import Proj, transform
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pylab as pl
from IPython import display
from cartopy import crs as ccrs, feature as cfeature
import cartopy.io.img_tiles as cimgt
from utide import solve, reconstruct

#%%
###############################################################################
########## LOAD LSPM CABLE  FEATURES : DISTANCE, LONGITUDE, LATITUDE, BATHY ...  + BATHYMETRY OF AREA OF STUDY #
###############################################################################
with open("/......./Interpolation_bathy_Nathan.csv",'r') as f :
    data_cable =  list(csv.reader(f, delimiter=","))
    
data_cable=np.array(data_cable)[1:,:]
inProj = Proj(init='epsg:2154')
outProj = Proj(init='epsg:4326')

longitude_cable = pd.Series([float(i) for i in data_cable[:,0]])
latitude_cable = pd.Series([float(i) for i in data_cable[:,1]])

longitude_cable, latitude_cable = transform(inProj,outProj,longitude_cable,latitude_cable)

distance_cable = pd.Series([float(i) for i in data_cable[:,3] ])
azimuth_cable = pd.Series([float(i) for i in data_cable[:,-1] ])
bathymetry_cable = pd.Series([np.nan if i == np.str_() else float(i) for i in data_cable[:,2] ])


longitude_cable = longitude_cable[::218]
latitude_cable = latitude_cable[::218]
distance_cable = distance_cable[::218]
azimuth_cable = azimuth_cable[::218]
bathymetry_cable = bathymetry_cable[::218]

longitude_cable= longitude_cable[(distance_cable>=0) & (distance_cable<=52500)]
latitude_cable= latitude_cable[(distance_cable>=0) & (distance_cable<=52500)]
azimuth_cable = azimuth_cable[(distance_cable>=0) & (distance_cable<=52500)]
bathymetry_cable = bathymetry_cable[(distance_cable>=0) & (distance_cable<=52500)]
distance_cable = distance_cable[(distance_cable>=0) & (distance_cable<=52500)]


request = cimgt.OSM()

# Parameters for Offshore section of LSPM cable
lon_inf = 5.85
lat_inf = 42.77
lon_sup = 6.04
lat_sup = 43.11
delta = 0.0012


# Bounds: (lon_min, lon_max, lat_min, lat_max):
extent = [lon_inf, lon_sup , lat_inf, lat_sup] 

######################## LOAD AND SLICE BATHYMETRIC DATA ######################
path_file = "/....../MNT_MED100m_GDL-CA_HOMONIM_WGS84_NM_ZNEG.grd"
## Reading in the netCDF file
data = xr.open_dataset(path_file)
longitude = data.variables['lon'][:].values
latitude = data.variables['lat'][:].values
bathymetry = data.variables['z'][:].values
lon_min = np.where(longitude>=lon_inf)[0].min()
lon_max = np.where(longitude<=lon_sup)[0].max()
longitude = longitude[lon_min:lon_max+1]
lat_min = np.where(latitude>=lat_inf)[0].min()
lat_max = np.where(latitude<=lat_sup)[0].max()
latitude = latitude[lat_min:lat_max+1]
bathymetry = bathymetry[lat_min:lat_max+1,lon_min:lon_max+1]



#%%
###############################################################################
########################### USEFUL FUNCTIONS TO ORGANIZE DATA ###########################
###############################################################################
def solve_and_predict(z, time, lat, dev=False, **kwargs):
    """ solve for tidal harmonics and create a tidal prediction
    """

    coef = solve(time, z, lat=lat)
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
    tide["h"]
    
    return ha, pr 

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
#%%
###############################################################################
######################### LOAD DAS-LF DATA FOR HARMONIC ANALYSIS ##########################
###############################################################################
path_file = "/....../DATA_HA_1000m.nc"      # <---- Choice : path DAS data

das_lf = xr.open_dataarray(path_file)

#%%
###############################################################################
################################ HARMONIC ANALYSIS ##################################
###############################################################################

HARMONIC_ANALYSIS = {}    

for i in range(len(das_lf.distance)):

    ha, pr = solve_and_predict(
        das_lf[i,:].values, 
        das_lf.time.values, 
        latitude_cable[i], constit = 'auto',       
)
    HARMONIC_ANALYSIS[str(i)] = ha.to_dataframe()
    

#%%    
###############################################################################
##################### PLOT HARMONIC ANALYSIS OF DAS DATA : A, A_CI OR SNR OR ENERGY ###############
###############################################################################

cste = 0
for i in list(HARMONIC_ANALYSIS.keys()): 
    
    fig, ax = plt.subplots(3,2,figsize=(32,8), subplot_kw={'projection': ccrs.PlateCarree()},gridspec_kw={'height_ratios': [1,0.8,0.5],"width_ratios":[0.8,1]})
    ############################## PLOT ###########################################

    gs = ax[0,1].get_gridspec()
    
    ax0 = fig.add_subplot(gs[:,0], projection=ccrs.PlateCarree())
    ax[0,0].remove()
    ax[1,0].remove()
    ax[2,0].remove()
    im = ax0.contourf(longitude,latitude,bathymetry,transform=ccrs.PlateCarree())
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
    ax0.plot(longitude_cable,latitude_cable,'k*',transform=ccrs.PlateCarree())
    ax0.plot(longitude_cable[cste],latitude_cable[cste],'r*',transform=ccrs.PlateCarree())
    ax0.legend(["LSPM cable","Channel position"],loc='best',fontsize=12)
    ax0.set_xlim((lon_inf,lon_sup))
    ax0.set_ylim((lat_inf,lat_sup))
    
    ax[0,1].remove()
    ax1 = fig.add_subplot(gs[0,1])
    p1 = ax1.plot((HARMONIC_ANALYSIS[str(i)]['omega_cpd'],HARMONIC_ANALYSIS[str(i)]['omega_cpd']),(np.zeros(len(HARMONIC_ANALYSIS[str(i)]['A'])), [j/max(HARMONIC_ANALYSIS[str(i)]['A']) for j in HARMONIC_ANALYSIS[str(i)]['A']]),c='black',linewidth=2)
   
    ax1.set_ylabel("Normalized Ampl. [-]",fontsize=16)
    
    for j in range(len(HARMONIC_ANALYSIS[str(i)]['omega_cpd'])):
        ax1.text(HARMONIC_ANALYSIS[str(i)]['omega_cpd'][j],HARMONIC_ANALYSIS[str(i)]['A'][j]/max(HARMONIC_ANALYSIS[str(i)]['A']),HARMONIC_ANALYSIS[str(i)].index[j],fontsize=15)

    p = ax1.plot(np.nan,'w')   
    ax1.legend(p,[f"High pass filter at 2 days , xi = {np.round(10**(-3)*das_lf.distance.values[cste],1)}km" ],loc="upper right", fontsize=15) 
    ax1.tick_params(axis='both', labelsize=16)
    ax1.grid()
    plt.setp(ax1.spines.values(), linewidth=3)
    
    
    ax[1,1].remove()
    ax2 = fig.add_subplot(gs[1,1])
    #x2 = [X for (X,Y) in sorted(zip(24*HARMONIC_ANALYSIS[str(i)]['omega_cpd'],HARMONIC_ANALYSIS[str(i)]['A_ci']))]
    #y2 = [Y for (X,Y) in sorted(zip(24*HARMONIC_ANALYSIS[str(i)]['omega_cpd'],HARMONIC_ANALYSIS[str(i)]['A_ci']))]
    #p2 ,= ax2.plot(x2,y2/max([max(HARMONIC_ANALYSIS[str(i)]['SNR']) for i in list(HARMONIC_ANALYSIS.keys())]),'*--b',linewidth=2)
    #p2 ,= ax2.plot(x2,y2,'*--b',linewidth=2)
    p2 = ax2. errorbar(HARMONIC_ANALYSIS[str(i)]['omega_cpd'], HARMONIC_ANALYSIS[str(i)]['A'],ms=5, yerr=[HARMONIC_ANALYSIS[str(i)]['A_ci'] ], capsize=2, fmt="ro", ecolor = "black")
    ax2.set_xlabel("Frequency [cpd]",fontsize=16)
    ax2.set_ylabel("DAS-LF [K]",fontsize=16)
    ax2.legend([p2],["Confidence interval 95%"],loc="upper right", fontsize=15) 
    ax2.tick_params(axis='both', labelsize=16)
   # ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax2.yaxis.get_offset_text().set_fontsize(18)
    #ax2.set_ylim((0,2*10**(-3)))
    ax2.grid()
    plt.setp(ax2.spines.values(), linewidth=3)
    
    ax[2,1].remove()
    ax3 = fig.add_subplot(gs[2,1])
    ax3.plot(10**(-3)*distance_cable,bathymetry_cable,'k-*', linewidth=3)
    ax3.set_xlabel("Distance [km]",fontsize=16)
    ax3.set_ylabel("Bathymetry [m]",fontsize=16)
    ax3.plot(10**(-3)*distance_cable.iloc[cste],bathymetry_cable.iloc[cste],'r*', linewidth=3)
    ax3.tick_params(axis='both', labelsize=16)
    ax3.grid()
    plt.setp(ax3.spines.values(), linewidth=3)
    
    fig.subplots_adjust(wspace=-0.2)
    gs = ax[2,1].get_gridspec()
    gs.update(hspace=0.5)
    plt.show()
    #fig.savefig(f'/......./.png')
    cste +=1
#%%
