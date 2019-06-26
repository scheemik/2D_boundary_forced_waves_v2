"""
A module to create a staircase background profile using tanh functions

Modified by Mikhail Schee, June 2019

"""

import numpy as np

# Functions to define an arbitrary density staircase profile
def lin_profile(z, z_b, z_t, val_b, val_t): # Steps can have a slope
    # Creates a linear profile from (z_b, val_b) to (z_t, val_t)
    values = 0*z
    slope = (val_t - val_b) / (z_t - z_b)
    values = slope*(z - z_b) + val_b
    return values
def val_ni(n, index, val_b, val_t): # interfaces are vertical
    i = index/2
    # returns value at interface i for n layers (bottom interface=0, bottom layer=1)
    return val_b + (i/n) * (val_t - val_b)
def staircase(z, n, ratio, z_b, z_t, val_b, val_t):
    # initialize array of values to be returned
    values = 0*z
    # find the thickness of the layers and the interfaces
    th_l = (z_t - z_b) / (n + (n-1)/ratio)
    th_i = th_l/ratio
    # z is an array of height values
    # function returns a corresponding array of values (density, for example)
    z_i = z_b
    index = 0 # even is interface, odd is layer
    # Loop from bottom z to top z, alternating layers and interfaces
    while (z_i < z_t):
        # Layer
        if (index%2 == 0):
            index += 1
            z_i += th_l
            val_below = val_ni(n, index-1, val_b, val_t)
            val_above = val_ni(n, index+1, val_b, val_t)
            values[(z>(z_i-th_l))&(z<z_i)] = lin_profile(z[(z>(z_i-th_l))&(z<z_i)], z_i-th_l, z_i, val_below, val_above)
        # Interface
        else:
            index += 1
            z_i += th_i
            values[(z>(z_i-th_i))&(z<z_i)] = val_ni(n, index, val_b, val_t)
    return values

def tanh_(z, bottom, top, slope, center):
    # initialize array of values to be returned
    values = 0*z
    # calculate height of step
    height = top - bottom
    # calculate step
    values = 0.5*height*(np.tanh(slope*(z-center))+1)+bottom
    return values
def rho_profile(z, n, bottom, top, slope, left, right):
    # initialize array of values to be returned
    values = 0*z
    # calculate height of domain
    H = top - bottom # don't take absolute value, this lets staircase flip
    # calculate height of steps
    height = H / float(n)
    # calculate width of domain
    W = abs(right - left)
    # calculate width of steps
    width = W / float(n)
    for i in range(n):
        b_i = i*height + bottom
        t_i = b_i + height
        c_i = right - (width/2.0 + i*width)
        values += tanh_(z, b_i, t_i, slope, c_i)
    return values

###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import time

from dedalus import public as de
from dedalus.extras import flow_tools

# Parameters
nx = 40*7
nz = 40*2
aspect_ratio = 4.0
Lx, Lz = (aspect_ratio, 1.)
z_b, z_t = (-Lz/2, Lz/2)

###############################################################################

# Create bases and domain
x_basis = de.Fourier('x', nx, interval=(-Lx/2, Lx/2), dealias=3/2)
z_basis = de.Chebyshev('z', nz, interval=(z_b, z_t), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)
# Initial conditions
x = domain.grid(0)
z = domain.grid(1)

###############################################################################

# Parameters to determine a specific staircase profile
n_layers = 5
slope = 50.0*n_layers
val_bot = +1.0
val_top = -1.0
z_bot = z_b
z_top = z_t

# Store profile in an array
bgpf_array = rho_profile(z, n_layers, val_bot, val_top, slope, z_bot, z_top)
#bgpf_array = rho_profile(z, n_layers, slope, z_b, z_t, val_bot, val_top)
#bgpf_array2 = staircase(z, n_layers, slope, z_b, z_t, val_bot, val_top)

# Plots the background profile
plot_bgpf = True
if (plot_bgpf):
    print(type(bgpf_array))
    #print(bgpf_array2)
    #print(bgpf_array2[0])
    plot_z = np.array(z[0])
    plot_b = np.array(bgpf_array[0])
    #hori2 = np.array(bgpf_array2[0])
    with plt.rc_context({'axes.edgecolor':'white', 'text.color':'white', 'axes.labelcolor':'white', 'xtick.color':'white', 'ytick.color':'white', 'figure.facecolor':'black'}):
        fg, ax1 = plt.subplots(1,1)
        #ax2 = ax1.twiny()

        ax1.set_title('Test Profile')
        ax1.set_ylabel(r'density ($\bar\rho$)')
        ax1.set_xlabel(r'depth ($z$)')
        ax1.plot(plot_z, plot_b, 'k-')

        #ax2.set_xlabel(r'frequency ($N^2$)')
        #ax2.plot(hori2, vert, '-')
        plt.grid(True)
        plt.show()
# %%

###############################################################################

# Functions to define an arbitrary N^2 staircase profile
#   N^2(z) = (g/rho_0)*(d rhobar/d z)
def N_lin_profile(z, z_b, z_t, val_b, val_t): # steps have N^2=const=slope
    # Creates a linear profile from (z_b, val_b) to (z_t, val_t)
    values = 0*z
    slope = (val_t - val_b) / (z_t - z_b)
    #values = slope*(z - z_b) + val_b
    values = values+slope
    print(values)
    return values #values
def N_val_ni(n, index, val_b, val_t): # interfaces go to N^2=0
    i = index/2
    # returns value at interface i for n layers (bottom interface=0, bottom layer=1)
    return val_b + (i/n) * (val_t - val_b)
def N_staircase(z, n, ratio, z_b, z_t, val_b, val_t):
    # initialize array of values to be returned
    values = 0*z
    # find the thickness of the layers and the interfaces
    th_l = (z_t - z_b) / (n + (n-1)/ratio)
    th_i = th_l/ratio
    # z is an array of height values
    # function returns a corresponding array of values (density, for example)
    z_i = z_b
    index = 0 # even is interface, odd is layer
    # Loop from bottom z to top z, alternating layers and interfaces
    while (z_i < z_t):
        # Layer
        if (index%2 == 0):
            index += 1
            z_i += th_l
            val_below = N_val_ni(n, index-1, val_b, val_t)
            val_above = N_val_ni(n, index+1, val_b, val_t)
            values[(z>(z_i-th_l))&(z<z_i)] = N_lin_profile(z[(z>(z_i-th_l))&(z<z_i)], z_i-th_l, z_i, val_below, val_above)
        # Interface
        else:
            index += 1
            z_i += th_i
            values[(z>(z_i-th_i))&(z<z_i)] = 0#N_val_ni(n, index, val_b, val_t)
    return values
