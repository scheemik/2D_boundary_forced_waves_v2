import matplotlib.pyplot as plt
import h5py
from dedalus import public as de
import numpy as np

#fig = plt.figure(figsize=(6, 10))
with h5py.File("_background_profile/current_bgpf/current_bgpf_s1.h5", mode='r') as file:
    N_0 = file['tasks']['N']
    N_0 = N_0[()]
    #print('N_0')
    #print(N_0[0][0])
    z = file['scales']['z']['1.0']
    z = z[()]
    #print('z')
    #print(z)
    #plt.plot(N_0[0][0], z, label="bgpf")
    #plt.ylim(0, 1.5)
    #plt.legend(loc='upper right', fontsize=10)
    #plt.show()

# Plotting function for sponge layer, background profile, etc.
def test_plot(vert, hori, plt_title, x_label, y_label, y_lims):
    #with plt.rc_context({'axes.edgecolor':'white', 'text.color':'white', 'axes.labelcolor':'white', 'xtick.color':'white', 'ytick.color':'white', 'figure.facecolor':'black'}):
    fg, ax = plt.subplots(1,1)
    ax.set_title(plt_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(y_lims)
    ax.plot(hori, vert, 'k-')
    plt.grid(True)
    return fg

plot_BP = True
rank = 0
LOC = True
z_b = -1.5
z_t =  0.0

# Plots the background profile
vert = np.array(z)
hori = np.array(N_0[0][0])
plt_title = 'Background Profile'
x_label = r'frequency ($N^2$)'
y_label = r'depth ($z$)'
y_lims  = [z_b,z_t]
fg = test_plot(vert, hori, plt_title, x_label, y_label, y_lims)
fg.savefig('test_plot.png')
if (plot_BP and rank == 0 and LOC):
    plt.show()
