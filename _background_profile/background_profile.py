"""
A module to create a staircase background profile

Modified by Mikhail Schee, June 2019

"""
import numpy as np
###############################################################################
# Functions to define an arbitrary density staircase profile

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
# Functions to define an arbitrary N^2 staircase profile

def cosh2(z, height, slope, center):
    # initialize array of values to be returned
    values = 0*z
    # calculate step
    values = (height*slope)/(2.0*(np.cosh(slope*(z-center)))**2.0)
    return values

def N2_profile(z, n, bottom, top, slope, left, right):
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
        c_i = right - (width/2.0 + i*width)
        values += cosh2(z, height, slope, c_i)
    return values
