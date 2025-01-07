# ------------------------------------------------
# Name: vortutils.py
# Author: Robby Frost
# University of Oklahoma
# Created: 21 June 2024
# Purpose: Functions for calculating vorticity 
# statistics from large eddy simulations of the
# convective boundary layer
# ------------------------------------------------

import xarray as xr
import numpy as np
import sys
import xrft
sys.path.append("/home/rfrost/LES-utils/")
sys.path.append("/home/rfrost/ABL_Transition/processing/")
from spec import autocorr_2d
from tranutils import ls_rot
from dask.diagnostics import ProgressBar

# ---------------------------------
# Settings to run functions
# ---------------------------------
# name of simulation
sim_label = "full_step_6"
# director for netCDF simulation output files
dnc = f"/home/rfrost/simulations/nc/{sim_label}/"
# first timestep
t0 = 576000
# final timestep
t1 = 580000
# number of timesteps between files
dt = 1000
# dimensional timestep (seconds)
delta_t = 0.05
# variable to calculate autocorr/roll factor
variable = "omega_z"
# number of angular/range bins for roll factor
ntbin = 21
nrbin = 20
# height index for calculations
hidx = 0

# flag to run calc_vorticity
calc_vort = False
# flag to run calc_vort_2d_autocorr
calc_vort_ac = False
# flag to run calc_vort_ls
calc_vort_ls = False
# flag to run roll_factor_vort
calc_vort_rf = False

# ---------------------------------
# Calculate the vorticity terms
# ---------------------------------
def calc_vorticity(dnc, t0, t1, dt, delta_t):
    """Calculate 4d vorticity statistics on netCDF LES output

    :param str dnc: absolute path to directory for saving output netCDF files
    :param int t0: first timestep for stats to be computed
    :param int t1: final timestep for stats to be computed
    :param int dt: number of timesteps between files to load
    :param float delta_t: dimensional timestep in simulation (seconds)
    """
    print("Beginning calc_vorticity.")

    # directories and configuration
    timesteps = np.arange(t0, t1+1, dt, dtype=np.int32)
    # determine files to read from timesteps
    fall = [f"{dnc}all_{tt:07d}.nc" for tt in timesteps]
    nf = len(fall)
    # calculate array of times represented by each file
    times = np.array([i*delta_t*dt for i in range(nf)])

    # Load files and clean up
    print("Reading files...")
    dd = xr.open_mfdataset(fall, combine="nested", concat_dim="time")
    dd.coords["time"] = times
    dd.time.attrs["units"] = "s"

    # Calculate statistics
    print("Beginning vorticity calculations")
    # create empty dataset that will hold everything
    dd_stat = xr.Dataset()

    # calculate horizontal vorticity
    dd_stat["omega_x"] = dd.w.differentiate(coord="y") - dd.v.differentiate(coord="z")
    dd_stat["omega_y"] = dd.w.differentiate(coord="x") - dd.u.differentiate(coord="z")
    dd_stat["omega_h"] = (dd_stat.omega_x ** 2 + dd_stat.omega_y ** 2) ** (1/2)
    # calculate vertical vorticity
    dd_stat["omega_z"] = dd.v.differentiate(coord="x") - dd.u.differentiate(coord="y")
    # calculate total vorticity
    dd_stat["omega_tot"] = (dd_stat.omega_x ** 2 + dd_stat.omega_y **2 + dd_stat.omega_z ** 2) ** (1/2)
    print("Finished vorticity components.")

    # 0-100 integrated stats
    dd_stat["0_100m_omega_x"] = dd_stat.omega_x[:,:,:,0:7].integrate("z")
    dd_stat["0_100m_omega_y"] = dd_stat.omega_y[:,:,:,0:7].integrate("z")
    dd_stat["0_100m_omega_h"] = dd_stat.omega_h[:,:,:,0:7].integrate("z")
    dd_stat["0_100m_omega_z"] = dd_stat.omega_z[:,:,:,0:7].integrate("z")
    dd_stat["0_100m_omega_tot"] = dd_stat.omega_tot[:,:,:,0:7].integrate("z")

    # vorticity averages
    dd_stat["omega_x_abs"] = abs(dd_stat.omega_x)
    dd_stat["omega_x_abs_mean"] = dd_stat.omega_x_abs.mean(dim=("x","y"))
    dd_stat["omega_y_abs"] = abs(dd_stat.omega_y)
    dd_stat["omega_y_abs_mean"] = dd_stat.omega_y_abs.mean(dim=("x","y"))
    dd_stat["omega_h_mean"] = dd_stat.omega_h.mean(dim=("x","y"))
    dd_stat["omega_z_abs"] = abs(dd_stat.omega_z)
    dd_stat["omega_z_abs_mean"] = dd_stat.omega_z_abs.mean(dim=("x","y"))
    dd_stat["omega_tot_mean"] = dd_stat.omega_tot.mean(dim=("x","y"))
    # positive vertical vorticity
    dd_stat["omega_z_pos"] = dd_stat.omega_z.where(dd_stat.omega_z > 0, drop=True)
    dd_stat["omega_z_pos_mean"] = dd_stat.omega_z_pos.mean(dim=("x","y"))

    print("Finished vorticity spatial averaging.")

    # Add attributes
    dd_stat.attrs = dd.attrs

    # save output
    fs_stat = f"{dnc}{t0}_{t1}_vort.nc"

    with ProgressBar():
        # save vorticity stats
        print(f"Saving file: {fs_stat}")
        dd_stat.to_netcdf(fs_stat, mode="w")

    print("Finished calc_vorticity! \n")
    return

# ---------------------------------
# Calculate vorticity 2d autocorr
# ---------------------------------
def calc_vort_2d_autocorr(dnc, t0, t1, variable, hidx):
    """Calculate 2d autocorrelation of vorticity variable using output from calc_vorticity

    :param str dnc: absolute path to directory for reading/saving output netCDF files
    :param int t0: first timestep for stats to be computed
    :param int t1: final timestep for stats to be computed
    :param str variable: variable to calculate autocorr on
    """
    print(f"Beginning calc_vort_2d_autocorr for {variable}.")

    # read in vorticity statistics
    dd_stat = xr.open_dataset(f"{dnc}{t0}_{t1}_vort.nc")
    dd_stat = dd_stat.isel(z=hidx)
    # dataset for autocorr data
    dd_ac = xr.Dataset()

    # 2d autocorrelation
    R_ds = autocorr_2d("naw", dd_stat, [variable], timeavg=False, output=False)
    dd_ac[f"{variable}_autocorr2d"] = R_ds[variable]

    print("Finished vorticity 2d autocorrelation.")

    # save output
    fs_ac = f"{dnc}{t0}_{t1}_{variable}_autocorr.nc"
    with ProgressBar():
        # save vorticity autocorrelation
        print(f"Saving file: {fs_ac}")
        dd_ac.to_netcdf(fs_ac, mode="w")

    print(f"Finished calc_vort_2d_autocorr on {variable}! \n")
    return

# ---------------------------------
# Calculate filtered vorticity 2d autocorr
# ---------------------------------
def calc_filter_vort_2d_autocorr(dnc, t0, t1, variable):
    """Calculate 2d autocorrelation of filtered vorticity using output from filter_vort

    :param str dnc: absolute path to directory for reading/saving output netCDF files
    :param int t0: first timestep for stats to be computed
    :param int t1: final timestep for stats to be computed
    :param str variable: variable to calculate autocorr on
    """
    print(f"Beginning calc_filter_vort_2d_autocorr for {variable}.")

    # read in vorticity statistics
    dd = xr.open_dataset(f"{dnc}{t0}_{t1}_vort_filtered.nc")
    # dd_stat = dd_stat.isel(z=0)
    # dataset for autocorr data
    dd_ac = xr.Dataset()

    # 2d autocorrelation
    R_ds = autocorr_2d("naw", dd, [variable], timeavg=False, output=False)
    dd_ac[f"{variable}_autocorr2d"] = R_ds[variable]

    print("Finished vorticity 2d autocorrelation.")

    # save output
    fs_ac = f"{dnc}{t0}_{t1}_{variable}_autocorr_filtered.nc"
    with ProgressBar():
        # save vorticity autocorrelation
        print(f"Saving file: {fs_ac}")
        dd_ac.to_netcdf(fs_ac, mode="w")

    print(f"Finished calc_filter_vort_2d_autocorr on {variable}! \n")
    return

# ---------------------------------
# Calculate vorticity length scales
# ---------------------------------
def calc_vort_length_scales(dnc, t0, t1):
    """Calculate integral length scales of vorticity using output from calc_vort_autocorr

    :param str dnc: absolute path to directory for reading/saving output netCDF files
    :param int t0: first timestep for stats to be computed
    :param int t1: final timestep for stats to be computed
    """
    print("Beginning calc_vort_length_scales.")

    # read in vorticity autocorr data
    R_ds = xr.open_dataset(f"{dnc}{t0}_{t1}_vort_autocorr.nc")
    # dataset for length scale data
    dd_ls = xr.Dataset()

    # integral length scales
    Lw_ds_omega_x = ls_rot(R_ds["omega_x_autocorr2d"], t0, t1, dt, read_in=False, output=False)
    dd_ls["ls_omega_x_rolls"] = Lw_ds_omega_x["rolls"]
    dd_ls["ls_omega_x_normal"] = Lw_ds_omega_x["normal"]
    Lw_ds_omega_y = ls_rot(R_ds["omega_y_autocorr2d"], t0, t1, dt, read_in=False, output=False)
    dd_ls["ls_omega_y_rolls"] = Lw_ds_omega_y["rolls"]
    dd_ls["ls_omega_y_normal"] = Lw_ds_omega_y["normal"]
    Lw_ds_omega_z = ls_rot(R_ds["omega_z_autocorr2d"], t0, t1, dt, read_in=False, output=False)
    dd_ls["ls_omega_z_rolls"] = Lw_ds_omega_z["rolls"]
    dd_ls["ls_omega_z_normal"] = Lw_ds_omega_z["normal"]
    print("Finished vorticity length scales.")

    # save output
    fs_ls = f"{dnc}{t0}_{t1}_vort_ls.nc"
    with ProgressBar():
        # save vorticity length scales
        print(f"Saving file: {fs_ls}")
        dd_ls.to_netcdf(fs_ls, mode="w")

    print("Finished calc_vort_length_scales! \n")
    return

# ---------------------------------
# Calculate vorticity roll factor
# ---------------------------------
def calc_roll_factor_vorticity(dnc, t0, t1, variable, ntbin=21, nrbin=20, hidx=0):
    """Calculate integral length scales of vorticity using output from calc_vort_autocorr

    :param str dnc: absolute path to directory for reading/saving output netCDF files
    :param int t0: first timestep for stats to be computed
    :param int t1: final timestep for stats to be computed
    :param str variable: variable to calculate roll factor for
    :param int ntbin: number of angular (theta) bins
    :param int nrbin: number of range bins
    :param int hidx: height index to calculate roll factor for
    """
    print("Beginning calc_roll_factor_vorticity.")

    # load in volumetric output
    df = xr.open_dataset(f"{dnc}{t0}_{t1}_stats.nc")
    Lx = max(df.x)

    # Convert to polar coords
    r = xr.open_dataset(f"{dnc}{t0}_{t1}_{variable}_autocorr.nc")
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
    w_pol.to_netcdf(f"{dnc}{t0}_{t1}_R_pol_{variable}.nc")

    # Calculate roll factor
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
    fsave = f"{dnc}{t0}_{t1}_roll_factor_{variable}.nc"
    # output to netCDF
    print(f"Saving file: {fsave}")
    with ProgressBar():
        roll.to_netcdf(fsave, mode="w")
    print("Finished!")

# ---------------------------------
# filtering code
# ---------------------------------
def filter_vorticity(dnc, t0, t1, hidx):
    """Filter total, horizontal, and 3 individual vorticity components at
    specified vertical level. Adapted from filter_vort.py.

    :param str dnc: absolute path to directory for reading/saving output netCDF files
    :param int t0: first timestep for stats to be computed
    :param int t1: final timestep for stats to be computed
    :param int hidx: height index to filter vorticity at
    """

    print("Beginning filter_vorticity.")

    # read in data
    df = xr.load_dataset(f"{dnc}{t0}_{t1}_vort.nc")
    variables = [key for key in df.variables.keys()]
    variables = variables[4:9]
    # dataset to save to
    Rsave = xr.Dataset(data_vars=None)
    
    # loop over vorticity variables
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

    return