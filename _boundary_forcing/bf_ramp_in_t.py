import matplotlib.pyplot as plt
import numpy as np

# Period of oscillation
T = 10.0
# Number of oscillations for ramp
n = 3
# Number of oscillations to plot
n_plt = n + 1
# Number of points to plot
n_pts = 100

# time array
t = np.linspace(0.0, n_plt*T, n_pts)

# ramp function
def ramp(t, n, T):
    vals = 0.5*( np.tanh((4.5/(n*T))*t - 2) + 1)
    return vals

fg, ax = plt.subplots(1,1)
ax.set_title('Ramp function')
ax.set_xlabel('t')
ax.set_ylabel('Amplitude')
ax.plot(t, ramp(t, n, T), 'k-')
plt.grid(True)
plt.show()
