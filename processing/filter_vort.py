# ------------------------------------------------
# Name: filter_vort.py
# Author: Robby Frost
# University of Oklahoma
# Created: 12 August 2024
# Purpose: Script to use a low-pass filter on
# vorticity from LES for computing roll factor
# ------------------------------------------------

import xarray as xr
import numpy as np
import xrft

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
t1 = 580000
# height index to filter at
hidx = 0

# ---------------------------------
# filtering code
# ---------------------------------
print("Beginning filter_vort.")

# read in data
df = xr.load_dataset(f"{dnc}{t0}_{t1}_vort.nc")
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
print("Finished filter_vort! \n")