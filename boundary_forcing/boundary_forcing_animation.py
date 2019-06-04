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
filename = 'boundary_forcing/' + name + '.gif'
anim.save(filename)# 'imageio')

plt.draw()
plt.show()

# %%
x, t, z, nu = symbols('x t z nu')
y = Function('y')
dsolve(Eq(y(t).diff(t,t) - y(t), exp(t)))

# %%
