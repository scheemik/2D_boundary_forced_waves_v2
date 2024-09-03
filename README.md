# 2D Boundary Forced Waves version 2

by Mikhail Schee

This is a repo to run 2D numerical experiments in Dedalus to reproduce the results of the laboratory wave tank experiment performed by Ghaemsaidi et al. (2016) "The impact of multiple layering on internal wave transmission," _Journal of Fluid Mechanics_, 789: 617–629. DOI: 10.1017/jfm.2015.682

I based the code off of the Dedalus v2 (as opposed to the current version, v3, as of writing) example of Rayleigh-Benard convection in 2 dimensions which can be found here:
https://github.com/DedalusProject/dedalus2
Look under `examples/ivp/2d_rayleigh_benard/rayleigh_benard.py`. It was from that script that I developed what is in the `current_code.py` script in this repo. Different versions of that script from different points in the project can be found in the `_code_checkpnts` folder.

This code uses a version of the Boussinesq equations, written here with the non-linear terms on the right-hand side as is required by this version of Dedalus:
* Mass conservation equation
    * $d_x(u) + d_z(w) = 0$
* Equation of state (in terms of buoyancy)
    * $d_t(b) - \kappa (d_x^2(b) + d_z^2(b)) = -w N^2(z) - (u d_x(b) + w d_z(b))$
        * Where $N(z)$ is the stratification profile (see `2D_exp_windows.png`)
* Horizontal momentum equation
    * $d_t(u) - S_L(z) \nu d_x^2(u) - \nu d_z^2(u) + d_x(p) = - (u d_x(u) + w d_z(u))$
        * Where $S_L(z)$ is the sponge layer (see `2D_exp_windows.png`)
* Vertical momentum equation
    * $d_t(w) - S_L(z) \nu d_x^2(w) - \nu d_z^2(w) + d_z(p) - b = - (u d_x(w) + w d_z(w))$

I set the boundary conditions for the Chebyshev basis, which is in the vertical, such that buoyancy, vertical velocity, and horizontal velocity are all zero at the bottom boundary such that no waves can originate there. At the top boundary, I construct the boundary conditions on those variables such that I produce a wave which satisfies the polarization relation and I limit the horizontal extent such that the result is a wave beam. I have constructed it in this way to as closey resemble the waves created by an oscillating cylinder at the top of the wave tank experiments by Ghaemsaidi et al. 2016.

The script I wrote to start each simulation is `run.sh` which accepts a number of arguments to modify what will happen in the experiment. These arguments are detailed in that script itself. I also wrote the `submit_job.sh` script which will send a job to the Niagara supercomputer from Compute Canada to run the `run.sh` script there.

The `VEF` files are from my attempt to recreate the vertical energy flux plots for my thesis. I found the values in those files by using software to analyze plots I'd made of the three components of vertical energy flux years ago and estimate the values. Take these with a grain of salt.