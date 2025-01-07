# ------------------------------------------------
# Name: roll_factor.py
# Author: Brian R. Greene and Robby Frost
# University of Oklahoma
# Created: 10 March 2023
# Purpose: Calculating 2d autocorrelation, 
# interpolating to a polar grid, and calculating
# roll factor.
# ------------------------------------------------
import sys
sys.path.append("/home/rfrost/LES-utils/")
# sys.path.append("/home/rfrost/ABL_Transition/processing/")

from spec import autocorr_2d
import xarray as xr
import numpy as np
import yaml
from dask.diagnostics import ProgressBar

# ---------------------------------
# Settings and read in data
# ---------------------------------
# read in volumetric simulation output
dnc = "/home/rfrost/simulations/nc/full_step_6/"
t0 = 576000
t1 = 580000
# set up bin centers for averaging
ntbin = 21
nrbin = 20
# height index
hidx = 0
# variable to calculate roll factor on
variable = "omega_z"
# flag to use filtered vorticity
filter_vort = True

# load in volumetric output
df = xr.open_dataset(f"{dnc}{t0}_{t1}_stats.nc")
Lx = max(df.x)

# ---------------------------------
# Convert to polar coords
# ---------------------------------
# read in autocorrelation dataset
if filter_vort:
    r = xr.open_dataset(f"{dnc}{t0}_{t1}_vort_autocorr_filtered.nc")
else:
    r = xr.open_dataset(f"{dnc}{t0}_{t1}_vort_autocorr.nc")
var = r[f"{variable}_autocorr2d"]#[:,:,:,hidx]
# grab sizes of x and y dimensions for looping
nx, ny, nt = var.x.size, var.y.size, var.time.size
x, y, time = var.x, var.y, var.time.values
r.close()

# calculate 2d arrays of theta=theta(x,y), r=r(x,y)
theta = np.arctan2(var.y, var.x)
r = (var.x**2. + var.y**2.) ** 0.5

rbin = np.linspace(0, Lx//2, nrbin)
tbin = np.linspace(-np.pi, np.pi, ntbin)
# intiialize empty arrays for storing values and counter for normalizing
wall, count = [np.zeros((nt, ntbin, nrbin), dtype=np.float64) for _ in range(2)]
# set up dimensional arrays to store roll factor stats
Rmax_r = np.zeros((nt, nrbin))
rbin_zi = np.zeros((nt, nrbin))
RR = np.zeros((nt))

print("Rotating to polar coordinates")
# loop over x, y pairs
for jx in range(nx):
    for jy in range(ny):
        # find nearest bin center for each r(jx,jy) and theta(jx,jy)
        jr = abs(rbin - r.isel(x=jx,y=jy).values).argmin()
        jt = abs(tbin - theta.isel(x=jx,y=jy).values).argmin()
        for t in range(nt):
            # store var[jt,jr] in wall, increment count
            wall[t,jt,jr] += var[t,jx,jy]
            count[t,jt,jr] += 1
# set up dimensial array for wmean
wmean = np.zeros((nt, ntbin, nrbin))
for t in range(nt):
    # normalize wall by count
    wmean[t,:,:] = wall[t,:,:] / count[t,:,:]

# convert polar Rww to xarray data array
w_pol = xr.DataArray(data=wmean,
                    coords=dict(time=time, theta=tbin, r=rbin),
                    dims=["time", "theta", "r"])
# output polar Rww data
if filter_vort:
    w_pol.to_netcdf(f"{dnc}{t0}_{t1}_R_pol_{variable}_filtered.nc")
else:
    w_pol.to_netcdf(f"{dnc}{t0}_{t1}_R_pol_{variable}.nc")

# ---------------------------------
# Calculate roll factor
# ---------------------------------
print("Calculating roll factor")
# loop over time
for t in range(nt):
    # calculate roll factor
    Rmax_r[t,:] = np.nanmax(wmean[t,:,:], axis=0) - np.nanmin(wmean[t,:,:], axis=0)
    rbin_zi[t,:] = rbin / df.z[hidx].values
    RR[t] = np.nanmax(Rmax_r[t, rbin_zi[t,:] >= 0.5])
print("Roll factor calculation complete!")
# create xarray data array
roll = xr.DataArray(data=RR,
                    coords=dict(time=time),
                    dims=["time"])

# save data
if filter_vort:
    fsave = f"{dnc}{t0}_{t1}_roll_factor_{variable}_filtered.nc"
else:
    fsave = f"{dnc}{t0}_{t1}_roll_factor_{variable}.nc"
# output to netCDF
print(f"Saving file: {fsave}")
with ProgressBar():
    roll.to_netcdf(fsave, mode="w")
print("Finished!")