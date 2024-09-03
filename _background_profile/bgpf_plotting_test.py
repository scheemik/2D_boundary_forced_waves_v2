"""
A script to test plotting the background profiles without
having to run the whole Dedalus script.

Modified by Mikhail Schee, June 2019

"""

###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

from dedalus import public as de
from dedalus.extras import flow_tools

import sys
sys.path.insert(0, './_background_profile')
from background_profile import rho_profile
from background_profile import N2_profile
from Foran_profile import Foran_profile
sys.path.insert(0, './_sponge_layer')
from sponge_layer import sponge_profile
# Add path to params files
sys.path.insert(0, './_params')
import lparams_local as lparams

# n_layers = 2
import params_repro1 as params
slope = params.profile_slope # 50.0*n_layers
str_bot = float(params.stair_bot) # -0.3
str_top = float(params.stair_top) # 0.3
import params_repro2 as params2
slope2 = params2.profile_slope # 50.0*n_layers
str_bot2 = float(params2.stair_bot) # -0.3
str_top2 = float(params2.stair_top) # 0.3

# Parameters
nx = lparams.n_x #512
nz = lparams.n_z #256
aspect_ratio = 4.0
Lx, Lz = float(params.L_x), float(params.L_z) # (aspect_ratio, 1.)
# z_b, z_t = (-Lz/2, Lz/2)
z_b, z_t = -Lz, 0.0

###############################################################################

# Create bases and domain
x_basis  = de.Fourier('x', nx, interval=params.x_interval, dealias=3/2)
z_main   = de.Chebyshev('zm', nz, interval=(z_b, z_t), dealias=3/2)
z_sponge = de.Chebyshev('zs', params.nz_sp, interval=(params.z_sb, z_b), dealias=3/2)
z_basis  = de.Compound('z', (z_sponge, z_main))
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)
# Initial conditions
x = domain.grid(0)
z = domain.grid(1)

###############################################################################

# Parameters to determine a specific staircase profile

z_bot = z_b
z_top = z_t
bump = (params.N_1 + params.N_2)/2
print('slope:', slope)
print('str_bot:', str_bot)
print('str_top:', str_top)

# Store profile in an array
#bgpf_array = rho_profile(z, n_layers, val_bot, val_top, slope, z_bot, z_top)
# bgpf_array2 = N2_profile(z, n_layers-1, params.N_0, float(params.stair_bot), float(params.stair_top), params.profile_slope, bump) 
bgpf_array1 = Foran_profile(z, 1-1, str_bot, str_top, slope, params.N_1, params.N_2)
bgpf_array2 = Foran_profile(z, 2-1, str_bot2, str_top2, slope2, params2.N_1, params2.N_2)
sponge_array = sponge_profile(z, params.z_sb, z_t, params.sp_slope, params.max_sp)

# Plots the background profile
plot_bgpf = True
if False:#(plot_bgpf):
    plot_z = np.array(z[0])
    #plot_p = np.array(bgpf_array[0])
    plot_N = np.array(bgpf_array2[0])
    # with plt.rc_context({'axes.edgecolor':'white', 'text.color':'white', 'axes.labelcolor':'white', 'xtick.color':'white', 'ytick.color':'white', 'figure.facecolor':'black'}):
    if True:
        fg, ax1 = plt.subplots(1,1)
        ax2 = ax1.twiny()

        # ax1.set_title('Test Profile')
        #ax1.set_xlabel(r'density ($\bar\rho$)')
        ax1.set_ylabel(r'depth ($z$)')
        ax1.set_ylim([z_b,z_t])
        #ax1.plot(plot_p, plot_z, 'k-')

        ax2.set_xlabel(r'frequency ($N^2$)')
        ax2.plot(plot_N, plot_z, '-')
        plt.grid(True)
        plt.show()


def set_fig_axes(heights, widths, fig_ratio=0.5, share_x_axis=None, share_y_axis=None):
    """
    Creates fig and axes objects based on desired heights and widths of subplots
    Ex: if widths=[1,5], there will be 2 columns, the 1st 1/5 the width of the 2nd

    heights     array of integers for subplot height ratios, len=rows
    widths      array of integers for subplot width  ratios, len=cols
    fig_ratio   ratio of height to width of overall figure
    share_x_axis bool whether the subplots should share their x axes
    share_y_axis bool whether the subplots should share their y axes
    """
    # Set aspect ratio of overall figure
    w, h = mpl.figure.figaspect(fig_ratio)
    # Find rows and columns of subplots
    rows = len(heights)
    cols = len(widths)
    # This dictionary makes each subplot have the desired ratios
    # The length of heights will be nrows and likewise len(widths)=ncols
    plot_ratios = {'height_ratios': heights,
                   'width_ratios': widths}
    # Determine whether to share x or y axes
    if share_x_axis == None and share_y_axis == None:
        if rows == 1 and cols != 1: # if only one row, share y axis
            share_x_axis = False
            share_y_axis = True
        elif rows != 1 and cols == 1: # if only one column, share x axis
            share_x_axis = True
            share_y_axis = False
        else:                       # otherwise, forget about it
            share_x_axis = False
            share_y_axis = False
    # Set ratios by passing dictionary as 'gridspec_kw', and share y axis
    return plt.subplots(figsize=(w,h), nrows=rows, ncols=cols, gridspec_kw=plot_ratios, sharex=share_x_axis, sharey=share_y_axis)

def plot_v_profiles(z_array, BP_array1, BP_array2, sp_array, omega1, omega2, filename='2D_exp_windows.png'):
    """
    Plots the vertical profiles: stratification, boundary forcing, sponge layer
        Note: all arrays must be imput as full-domain, no trimmed versions

    z_array     1D array of z values
    BP_array    1D array of the background profile values in z
    bf_array    1D array of boundary forcing window
    sp_array    1D array of sponge layer window
    kL          Non-dimensional number relating wavelength and layer thickness
    theta       Angle at which wave is incident on stratification structure
    omega       frequency of wave
    z_I         depth to measure incident wave
    z_T         depth to meausre transmitted wave
    z0_dis      top of vertical structure extent
    zf_dis      bottom of vertical structure extent
    plot_full...True or False, include depths outside display and transient period?
    """
    # Set figure and axes for plot
    fig, axes = set_fig_axes([1], [1,1,1], 0.75)
    # Plot the sponge layer function
    axes[0].plot(sp_array, z_array, color=my_clrs['F_sp'], linestyle=':')
    axes[0].set_xlabel(r'Amplitude')
    axes[0].set_title(r'Sponge layer')
    # Plot the background profile of N_0 for a single layer
    axes[1].plot(BP_array1, z_array, color=my_clrs['N_0'], label=r'$N_1(z)$')
    axes[1].set_xlabel(r'$N$ (s$^{-1}$)')
    axes[1].set_title(r'Single layer')
    # Add vertical line for the value of omega
    axes[1].axvline(x=omega1, color=my_clrs['omega'], linestyle=':', label=r'$\omega_1$')
    axes[1].legend()
    # Plot the background profile of N_0 for a double layer
    axes[2].plot(BP_array2, z_array, color=my_clrs['N_0'], label=r'$N_2(z)$')
    axes[2].set_xlabel(r'$N$ (s$^{-1}$)')
    axes[2].set_title(r'Double layer')
    # Add vertical line for the value of omega
    axes[2].axvline(x=omega2, color=my_clrs['omega'], linestyle=':', label=r'$\omega_2$')
    axes[2].legend()
    # Set y-axis labels
    axes[0].set_ylabel(r'$z$ (m)')
    # Set plot bounds
    axes[0].set_ylim([-1.5, 0])
    axes[1].set_ylim([-1.5, 0])
    axes[2].set_ylim([-1.5, 0])
    # Add horizontal lines
    axes[0].axhline(y=-0.5, color=my_clrs['black'], linestyle='--')
    axes[1].axhline(y=-0.5, color=my_clrs['black'], linestyle='--')
    axes[2].axhline(y=-0.5, color=my_clrs['black'], linestyle='--')
    if not isinstance(filename, type(None)):
        plt.savefig(filename, dpi=300)
    else:
        plt.show()

###############################################################################
# Plotting colors from style guide

CUSTOM_COLORS ={'lightcornflowerblue2': '#a4c2f4',
                'lightred3': '#f2c1c1'}

TAB_COLORS =   {'tab:blue': '#1f77b4',
                'tab:orange': '#ff7f0e',
                'tab:green': '#2ca02c',
                'tab:red': '#d62728',
                'tab:purple': '#ffffff',
                'tab:brown': '#ffffff',
                'tab:pink': '#ffffff',
                'tab:gray': '#ffffff',
                'tab:olive': '#ffffff',
                'tab:cyan': '#ffffff'
                }

CSS4_COLORS =  {'aliceblue': '#F0F8FF',
                'antiquewhite': '#FAEBD7',
                'aqua': '#00FFFF',
                'aquamarine': '#7FFFD4',
                'azure': '#F0FFFF',
                'beige': '#F5F5DC',
                'bisque': '#FFE4C4',
                'black': '#000000',
                'blanchedalmond': '#FFEBCD',
                'blue': '#0000FF',
                'blueviolet': '#8A2BE2',
                'brown': '#A52A2A',
                'burlywood': '#DEB887',
                'cadetblue': '#5F9EA0',
                'chartreuse': '#7FFF00',
                'chocolate': '#D2691E',
                'coral': '#FF7F50',
                'cornflowerblue': '#6495ED',
                'cornsilk': '#FFF8DC',
                'crimson': '#DC143C',
                'cyan': '#00FFFF',
                'darkblue': '#00008B',
                'darkcyan': '#008B8B',
                'darkgoldenrod': '#B8860B',
                'darkgray': '#A9A9A9',
                'darkgreen': '#006400',
                'darkgrey': '#A9A9A9',
                'darkkhaki': '#BDB76B',
                'darkmagenta': '#8B008B',
                'darkolivegreen': '#556B2F',
                'darkorange': '#FF8C00',
                'darkorchid': '#9932CC',
                'darkred': '#8B0000',
                'darksalmon': '#E9967A',
                'darkseagreen': '#8FBC8F',
                'darkslateblue': '#483D8B',
                'darkslategray': '#2F4F4F',
                'darkslategrey': '#2F4F4F',
                'darkturquoise': '#00CED1',
                'darkviolet': '#9400D3',
                'deeppink': '#FF1493',
                'deepskyblue': '#00BFFF',
                'dimgray': '#696969',
                'dimgrey': '#696969',
                'dodgerblue': '#1E90FF',
                'firebrick': '#B22222',
                'floralwhite': '#FFFAF0',
                'forestgreen': '#228B22',
                'fuchsia': '#FF00FF',
                'gainsboro': '#DCDCDC',
                'ghostwhite': '#F8F8FF',
                'gold': '#FFD700',
                'goldenrod': '#DAA520',
                'gray': '#808080',
                'green': '#008000',
                'greenyellow': '#ADFF2F',
                'grey': '#808080',
                'honeydew': '#F0FFF0',
                'hotpink': '#FF69B4',
                'indianred': '#CD5C5C',
                'indigo': '#4B0082',
                'ivory': '#FFFFF0',
                'khaki': '#F0E68C',
                'lavender': '#E6E6FA',
                'lavenderblush': '#FFF0F5',
                'lawngreen': '#7CFC00',
                'lemonchiffon': '#FFFACD',
                'lightblue': '#ADD8E6',
                'lightcoral': '#F08080',
                'lightcyan': '#E0FFFF',
                'lightgoldenrodyellow': '#FAFAD2',
                'lightgray': '#D3D3D3',
                'lightgreen': '#90EE90',
                'lightgrey': '#D3D3D3',
                'lightpink': '#FFB6C1',
                'lightsalmon': '#FFA07A',
                'lightseagreen': '#20B2AA',
                'lightskyblue': '#87CEFA',
                'lightslategray': '#778899',
                'lightslategrey': '#778899',
                'lightsteelblue': '#B0C4DE',
                'lightyellow': '#FFFFE0',
                'lime': '#00FF00',
                'limegreen': '#32CD32',
                'linen': '#FAF0E6',
                'magenta': '#FF00FF',
                'maroon': '#800000',
                'mediumaquamarine': '#66CDAA',
                'mediumblue': '#0000CD',
                'mediumorchid': '#BA55D3',
                'mediumpurple': '#9370DB',
                'mediumseagreen': '#3CB371',
                'mediumslateblue': '#7B68EE',
                'mediumspringgreen': '#00FA9A',
                'mediumturquoise': '#48D1CC',
                'mediumvioletred': '#C71585',
                'midnightblue': '#191970',
                'mintcream': '#F5FFFA',
                'mistyrose': '#FFE4E1',
                'moccasin': '#FFE4B5',
                'navajowhite': '#FFDEAD',
                'navy': '#000080',
                'oldlace': '#FDF5E6',
                'olive': '#808000',
                'olivedrab': '#6B8E23',
                'orange': '#FFA500',
                'orangered': '#FF4500',
                'orchid': '#DA70D6',
                'palegoldenrod': '#EEE8AA',
                'palegreen': '#98FB98',
                'paleturquoise': '#AFEEEE',
                'palevioletred': '#DB7093',
                'papayawhip': '#FFEFD5',
                'peachpuff': '#FFDAB9',
                'peru': '#CD853F',
                'pink': '#FFC0CB',
                'plum': '#DDA0DD',
                'powderblue': '#B0E0E6',
                'purple': '#800080',
                'rebeccapurple': '#663399',
                'red': '#FF0000',
                'rosybrown': '#BC8F8F',
                'royalblue': '#4169E1',
                'saddlebrown': '#8B4513',
                'salmon': '#FA8072',
                'sandybrown': '#F4A460',
                'seagreen': '#2E8B57',
                'seashell': '#FFF5EE',
                'sienna': '#A0522D',
                'silver': '#C0C0C0',
                'skyblue': '#87CEEB',
                'slateblue': '#6A5ACD',
                'slategray': '#708090',
                'slategrey': '#708090',
                'snow': '#FFFAFA',
                'springgreen': '#00FF7F',
                'steelblue': '#4682B4',
                'tan': '#D2B48C',
                'teal': '#008080',
                'thistle': '#D8BFD8',
                'tomato': '#FF6347',
                'turquoise': '#40E0D0',
                'violet': '#EE82EE',
                'wheat': '#F5DEB3',
                'white': '#FFFFFF',
                'whitesmoke': '#F5F5F5',
                'yellow': '#FFFF00',
                'yellowgreen': '#9ACD32'}

my_clrs       =  {'b': TAB_COLORS['tab:blue'],
                  'w': (1, 0, 0),               # - r
                  'u': (0, 0, 1),               # - b
                  'v'  : (0, 0.5, 0),           # - g
                  'p': CSS4_COLORS['plum'],
                  'diffusion': CSS4_COLORS['peru'],
                  'viscosity': CSS4_COLORS['peru'],
                  'N_0': "#490092",  #  5 dark purple # 'skyblue',
                  'rho': CSS4_COLORS['slateblue'],
                  'advection': CSS4_COLORS['indianred'],
                  'coriolis': CSS4_COLORS['teal'],
                  'omega': "#006ddb",  #  6 royal blue #  'lightcoral',
                  'F_bf': "#920000",  # 10 dark red # '#008080',            # - teal
                  'F_sp': "#004949",  #  1 dark olive # '#CD853F',            # - peru
                  'temperature': '#B22222',     # - firebrick
                  'salinity': '#4682B4',        # - steelblue
                  'incident': "#db6d00",  # 12 dark orange # '#8A2BE2',        # - blueviolet
                  'reflection': '#4169E1',      # - royalblue
                  'transmission': "#009292",  #  2 teal # '#FF6347',    # - tomato
                  'linear': CSS4_COLORS['forestgreen'],
                  'nonlinear': CSS4_COLORS['indianred'],
                  'arctic': CSS4_COLORS['cornflowerblue'],
                  'cold-fresh': CUSTOM_COLORS['lightcornflowerblue2'],
                  'warm-salty': CUSTOM_COLORS['lightred3'],
                  'black': (0, 0, 0),
                  'white': (1, 1, 1)}

plot_v_profiles(np.array(z[0]), np.array(bgpf_array1[0]), np.array(bgpf_array2[0]), np.array(sponge_array[0]), params.omega, params2.omega)#, filename='f_1D_windows.png')