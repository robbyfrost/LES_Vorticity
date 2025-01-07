# ------------------------------------------------
# Name: roll_factor_filter.py
# Author: Robby Frost
# University of Oklahoma
# Created: 13 August 2024
# Purpose: Filter vorticity variables, calculate 
# autocorrelation of filtered variables, and 
# calculate roll factor of filtered variables.
# ------------------------------------------------

import xarray as xr
import numpy as np
import xrft
from vortutils import calc_filter_vort_2d_autocorr

# ---------------------------------
# Settings
# ---------------------------------
# name of simulation
sim_label = "full_step_6"
# director for netCDF simulation output files
dnc = f"/home/rfrost/simulations/nc/{sim_label}/"
# first timestep
t0 = 576000
# final timestep
t1 = 1152000
# number of timesteps between files
dt = 1000
# dimensional timestep (seconds)
delta_t = 0.05
# height index to filter at
hidx = 0
# set up bin centers for roll factor averaging
ntbin = 21
nrbin = 20

# ---------------------------------
# filtering vorticity
# ---------------------------------
print("Beginning filtering code.")

# read in data
df = xr.open_dataset(f"{dnc}{t0}_{t1}_vort.nc")
variables = [key for key in df.variables.keys()]
variables = variables[4:9]
# dataset to save to
Rsave = xr.Dataset(data_vars=None)

for v in variables:
    try:
        var = df[v].isel(z=hidx)
    except ValueError:
        continue
    # take fft
    f_var = xrft.fft(var, dim=('x','y'), true_phase=True, true_amplitude=True)
    # take cutoff wavenumber as 1/1000 /m
    fc = 1/1000.

    # zero out x and y wavenumbers above this cutoff to get lowpass filter
    jxp = np.where(f_var.freq_x > fc)[0]
    jxn = np.where(f_var.freq_x < -fc)[0]
    jyp = np.where(f_var.freq_y > fc)[0]
    jyn = np.where(f_var.freq_y < -fc)[0]
    f_var[:,jxp,:] = 0
    f_var[:,jxn,:] = 0
    f_var[:,:,jyp] = 0
    f_var[:,:,jyn] = 0

    # take ifft
    var_filt = xrft.ifft(f_var, dim=('freq_x','freq_y'),
                        true_amplitude=True, true_phase=True,
                        lag=(f_var.freq_x.direct_lag, f_var.freq_y.direct_lag)).real
    # add to dataset
    Rsave[v] = var_filt

print("Calculation complete.")

# output filtered dataset
dout = f"{dnc}{t0}_{t1}_vort_filtered.nc"
print(f"Saving file: {dout}")
Rsave.to_netcdf(dout)
print("Finished filtering! \n")

# ---------------------------------
# filtered 2D autocorrelation
# ---------------------------------
for v in variables:
    calc_filter_vort_2d_autocorr(dnc, t0, t1, v)

# ---------------------------------
# roll factor
# ---------------------------------
print("Beginning roll factor.")
for v in variables:
    print(f"Calculating for {v}.")
    # load in volumetric output
    df = xr.open_dataset(f"{dnc}{t0}_{t1}_stats.nc")
    Lx = max(df.x)

    # Convert to polar coords
    r = xr.open_dataset(f"{dnc}{t0}_{t1}_{v}_autocorr_filtered.nc")
    var = r[f"{v}_autocorr2d"]#[:,:,:,hidx]
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

    print("Rotating to polar coordinates.")
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
    w_pol.to_netcdf(f"{dnc}{t0}_{t1}_R_pol_{v}_filtered.nc")

    # Calculate roll factor
    print("Calculating roll factor.")
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
    fsave = f"{dnc}{t0}_{t1}_roll_factor_{v}_filtered.nc"
    # output to netCDF
    print(f"Saving file: {fsave}")

    roll.to_netcdf(fsave, mode="w")
    print(f"Finished roll factor for {v}! \n")

print("Script complete :) \n")