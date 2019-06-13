"""
A script for visualizing by boundary forcing functions

"""

!pip3 install --upgrade
!pip3 install sympy
!pip3 install matplotlib

# %%
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sympy import *

# %%
# this one is an example from a webpage and it works
fig, ax = plt.subplots(figsize=(5, 3))
ax.set(xlim=(-3, 3), ylim=(-1, 1))
x = np.linspace(-3, 3, 91)
t = np.linspace(1, 25, 30)
X2, T2 = np.meshgrid(x, t)

sinT2 = np.sin(2*np.pi*T2/T2.max())
F = 0.9*sinT2*np.sinc(X2*(1 + sinT2))

line = ax.plot(x, F[0, :], color='k', lw=2)[0]

def animate(i):
    line.set_ydata(F[i, :])

anim = FuncAnimation(
    fig, animate, interval=100, frames=len(t)-1)

name = 'test'
filename = '_boundary_forcing/' + name + '.gif'
anim.save(filename)# 'imageio')

#plt.draw()
#plt.show()

# %%
# this is my own modification
x_min = -3
L_x = 6
x_max = x_min + L_x

z = 0
kz = 1
kx = 1
omega = .2
period = 2*np.pi/omega

fig, ax = plt.subplots(figsize=(5, 3))
ax.set(xlim=(x_min, x_max), ylim=(-1, 1))
x = np.linspace(x_min, x_max, 91)
t = np.linspace(0, period, 30)
X2, T2 = np.meshgrid(x, t)

# Windowing function (gaussian)
a = 1.0     # height of peak
b = 0.0     # center of peak
# FWHM = 2\sqrt{2*ln{2}}*c
FWHM = L_x/6
c = FWHM / (2*np.sqrt(2*np.log(2)))     # RMS width
win = a*np.exp(-(X2-b)**2/(2.0*c)**2)

# Boundary forcing for u
Fu = -np.sin(kx*X2 + kz*z - omega*T2)*win
# Boundary forcing for w
Fw =  np.sin(kx*X2 + kz*z - omega*T2)*win
# Boundary forcing for b
Fb =  np.cos(kx*X2 + kz*z - omega*T2)*win

line0 = ax.plot(x, win[0, :], '--', lw=1)[0]
line1 = ax.plot(x, Fu[0, :], color='r', lw=2)[0]
line2 = ax.plot(x, Fw[0, :], color='k', lw=2)[0]
line3 = ax.plot(x, Fb[0, :], color='b', lw=2)[0]

def animate(i):
    line0.set_ydata(win[i, :])
    line1.set_ydata(Fu[i, :])
    line2.set_ydata(Fw[i, :])
    line3.set_ydata(Fb[i, :])

anim = FuncAnimation(
    fig, animate, interval=period, frames=len(t)-1)

name = 'test'
filename = '_boundary_forcing/' + name + '.gif'
anim.save(filename)

#plt.draw()
#plt.show()

# %%
x, t, z, nu = symbols('x t z nu')
y = Function('y')
dsolve(Eq(y(t).diff(t,t) - y(t), exp(t)))

# %%
