# ------------------------------------------------
# Name: run_vortutils.py
# Author: Robby Frost
# University of Oklahoma
# Created: 13 August 2024
# Purpose: Script to run functions from vortutils.py
# on non-filtered data
# ------------------------------------------------

from vortutils import *

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
calc_vort_rf = True

# ---------------------------------
# Run desired scripts
# ---------------------------------
if calc_vort:
    calc_vorticity(dnc, t0, t1, dt, delta_t)

if calc_vort_ac:
    calc_vort_2d_autocorr(dnc, t0, t1, variable, hidx)

if calc_vort_ls:
    calc_vort_length_scales(dnc,t0,t1)

if calc_vort_rf:
    calc_roll_factor_vorticity(dnc, t0, t1, variable, ntbin, nrbin, hidx)