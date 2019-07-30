import matplotlib.pyplot as plt
import h5py
from dedalus import public as de

fig = plt.figure(figsize=(6, 10))
with h5py.File("_background_profile/current_bgpf/current_bgpf_s1.h5", mode='r') as file:
    N_0 = file['tasks']['N']
    N_0 = N_0[()]
    print('N_0')
    print(N_0[0][0])
    z = file['scales']['z']['1.0']
    z = z[()]
    print('z')
    print(z)
    plt.plot(N_0[0][0], z, label="bgpf")
    #plt.ylim(0, 1.5)
    #plt.legend(loc='upper right', fontsize=10)
    plt.show()
