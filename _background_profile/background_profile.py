"""
A module to create a staircase background profile

Modified by Mikhail Schee, June 2019

"""
import numpy as np
###############################################################################
# Functions to define an arbitrary density staircase profile

def tanh_(z, height, slope, center):
    # initialize array of values to be returned
    values = 0*z
    # calculate step
    values = 0.5*height*(np.tanh(slope*(z-center))+1)
    return values

def cosh2(z, height, slope, center):
    # initialize array of values to be returned
    values = 0*z
    # calculate step
    values = height/(np.cosh(slope*(z-center))**2.0)
    #values = (height*slope)/(2.0*(np.cosh(slope*(z-center)))**2.0)
    return values

def rho_profile(z, n, bottom, top, slope, left, right):
    # initialize array of values to be returned
    values = 0*z
    # calculate height of staircase in value (density)
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
# Functions to define an arbitrary N^2 staircase profile

def N2_profile(z, n, bg_height, stair_bot, stair_top, slope, bump):
    # initialize array of values to be returned
    values = 0*z
    # Add upper stratification
    values += tanh_(z, bg_height, slope, stair_top)
    # Add lower stratification
    values += tanh_(z, bg_height, -slope, stair_bot)
    # Find height of staircase region
    H = stair_top - stair_bot
    # If there are steps to be added...
    if (n > 0):
        # calculate height of steps
        height = H / float(n+1)
        for i in range(n):
            c_i = stair_bot + (i+1)*height
            values += cosh2(z, bump, slope, c_i)
    return values
