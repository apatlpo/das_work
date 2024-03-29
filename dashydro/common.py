import os
import pandas as pd

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.geodesic as cgeo

# ------------------------------- paths ----------------------------------------

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


deployments_info = {
    "2019_summer_toulon": dict(start = pd.Timestamp(2019,7,13),
                               end = pd.Timestamp(2019,7,31),
                               site = "toulon",
                               ),
    "2020_spring_toulon": dict(start = pd.Timestamp(2020,4,13),
                               end = pd.Timestamp(2020,5,12),
                               site = "toulon",
                               ),
    "2020_autumn_toulon": dict(start = pd.Timestamp(2020,10,16),
                               end = pd.Timestamp(2020,11,5),
                               site = "toulon",
                               ),
    "2021_winter_toulon": dict(start = pd.Timestamp(2020,12,11),
                               end = pd.Timestamp(2021,1,6),
                               site = "toulon",
                               ),
    "2021_spring_monaco": dict(start = pd.Timestamp(2021,4,30),
                               end = pd.Timestamp(2021,6,9),
                               site = "monaco",
                               ),
    "2022_summer_toulon": dict(start = pd.Timestamp(2022,6,3),
                               end = pd.Timestamp(2022,6,13),
                               site = "toulon",
                               ),
}

for deployment in deployments_info:
    deployments_info[deployment]["raw_dir"] = raw_dir(deployment)
    deployments_info[deployment]["processed_dir"] = processed_dir(deployment)

geo_domains = dict(toulon=dict(lon = slice(5, 7),
                              lat = slice(42, 43.5),
                              ),
                  monaco=dict(lon = slice(7., 8.5),
                              lat = slice(43.3, 44.2),
                              )
                  )

# -------------------------- plotting ------------------------------------------

def plot_map(da,
             site = "toulon",
             extent = None,
             figsize=(10,7),
             **kwargs,
             ):

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection=ccrs.Orthographic(6, 43.))

    da.plot(ax=ax, transform=ccrs.PlateCarree(), **kwargs)

    if extent is None:
        gd = geo_domains[site]
        extent = [gd["lon"].start, gd["lon"].stop, gd["lat"].start, gd["lat"].stop]
    ax.set_extent(extent)

    gl = ax.gridlines(draw_labels=True, dms=False,
                      x_inline=False, y_inline=False,
                    )
    gl.right_labels=False
    gl.top_labels=False
    ax.coastlines(resolution='10m', color="orange", lw=3, linestyle='-', alpha=1)
    ax.add_feature(cfeature.LAND, zorder=2)

    return fig, ax
