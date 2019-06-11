"""
A module to create a staircase background profile

Modified by Mikhail Schee, June 2019

"""
# Functions to define an arbitrary staircase profile
def lin_profile(z, z_b, z_t, val_b, val_t):
    # Creates a linear profile from (z_b, val_b) to (z_t, val_t)
    values = 0*z
    slope = (val_t - val_b) / (z_t - z_b)
    values = slope*(z - z_b) + val_b
    return values
def val_ni(n, index, val_b, val_t):
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
