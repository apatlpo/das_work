# Install required libraries for project :

## install miniconda (allways required)

Download Miniconda3 (i.e. for python3) from the [conda website](https://conda.io/miniconda.html)

```
bash Miniconda3-latest-Linux-x86_64.sh
bash
conda update conda
```

## install from scratch

```
conda create -n dashydro -c conda-forge python=3.8 tqdm \
            xarray zarr netcdf4 h5py dask-jobqueue bottleneck \
            jupyterlab ipywidgets \
            cartopy geopandas descartes xesmf \
            datashader geoviews hvplot \
            scikit-learn seaborn \
            xgcm cmocean gsw pyinterp \
            xhistogram xrft

conda activate dashydro

pip install git+https://github.com/apatlpo/synthetic_stats.git
pip install git+https://github.com/psf/black

# may need to ask permissions to aurelien here
git clone https://github.com/apatlpo/das_work.git
cd dashydro; pip install -e .
cd ..

# may need to ask permissions to aurelien here
git clone https://github.com/apatlpo/pynsitu.git
cd pynsitu; pip install -e .
cd ..

```

### Uninstall library with `pip install -e .`:

- remove the egg file ( `print(distributed.__file__)` for example)
- from file `easy-install.pth`, remove the corresponding line (it should be a path to the source directory or of an egg file).

## install from environment file

```
conda env create --file environment.yml
# some install still need to be done manually
pip install -e .      # installs the local library
pip install git+https://github.com/apatlpo/synthetic_stats.git
```

In order to create an environment file, you need to run:

```
conda env export --name dashydro --file environment.yml
#--from-history
```

# General information about miniconda:

## Overview

Miniconda installers contain the conda package manager and Python.
Once miniconda is installed, you can use the conda command to install any other packages and create environments.

After downloading `Miniconda3-latest-Linux-x86_64.sh` or `Miniconda3-latest-MacOSX-x86_64.sh` you need to run it with: `bash Miniconda3-latest-MacOSX-x86_64.sh`

Miniconda must be used with bash. If you want to use it with csh, add in your .cshrc:
```
#
#----------------------------------------------------------------
# alias Miniconda
#----------------------------------------------------------------
#
source $home/.miniconda3/etc/profile.d/conda.csh
conda activate dashydro
```

## Main commands:
What version, update conda
```
conda --version
conda update conda
```
Create new environment dashydro
```
conda create --name dashydro python
```
Switch to another environment (activate/deactivate) (or source_activate in csh)
```
conda activate dashydro
```
To change your path from the current environment back to the root (or source_deactivate in csh)
```
conda deactivate
```
List all environments
```
conda env list
```
Delete an environment
```
conda remove --name dashydro --all
```
View a list of packages and versions installed in an environmentSearch for a package
```
conda list
```
Check to see if a package is available for conda to install
```
conda search packagename
```
Install a new package
```
conda install packagename
```
Remove one library (and not dependencies):
```
conda remove thislibrary --force
```
Remove unused packages and caches:
```
conda clean --all
```
Remove conda
```
rm -rf /home/machine/username/miniconda3
```
where machine is the name of your computer and username is your username.


## Install a package from Anaconda.org

For packages that are not available using conda install, we can next look on Anaconda.org. Anaconda.org is a package management service for both public and private package repositories. Anaconda.org is a Continuum Analytics product, just like Anaconda and Miniconda.

In a browser, go to http://anaconda.org. We are looking for a package named “pestc4py”
There are more than a dozen copies of petsc4py available on Anaconda.org, first select your platform, then you can sort by number of downloads by clicking the “Downloads” heading.

Select the version that has the most downloads by clicking the package name. This brings you to the Anaconda.org detail page that shows the exact command to use to download it:

Check to see that the package downloaded
```
conda list
```

## Install a package with pip

For packages that are not available from conda or Anaconda.org, we can often install the package with pip (short for “pip installs packages”).
Exporting environment



Old stuff:

```
conda env export > environment.yml  #on a machine
conda env create -f environment.yml -n $ENV_NAME  #on the new machine
```
