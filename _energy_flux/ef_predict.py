# A script to predit the energy flux components from the parameters of an experiment

import numpy as np

# Define the parameters
g = 9.81
nu = 1.0e-6
A_hat = 2.3e-4

const_N_0   = 1.0
const_omega = 0.7071
lam_x = 0.5 / 3.0
const_k_x   = 2*np.pi/lam_x # [m^-1] k*cos(theta)
const_N_params = {
    'N_0': const_N_0,                       # [rad s^-1]    
    'omega': const_omega,                   # [rad s^-1]
    'k': const_k_x*const_N_0/const_omega    # [m^-1]
}
single_layer_params = {
    'N_0': 0.95,
    'omega': 0.67,
    'k': 45,
}
double_layer_params = {
    'N_0': 0.95,
    'omega': 0.72,
    'k': 45,
}

# The predictions for energy flux componenets based on the polarization relation
def make_ef_predictions(params):
    N_0 = params['N_0']
    omega = params['omega']
    k = params['k']
    theta = np.arccos(omega/N_0)    # [rad]
    k_x   = k*omega/N_0             # [m^-1] k*cos(theta)
    k_z   = k*np.sin(theta)         # [m^-1] k*sin(theta)
    c_gz  = omega * k_z / (k_x**2 + k_z**2)
    # number of oscillations to get from top to bottom
    n_T = 0.5 / (c_gz * 2*np.pi / omega)
    # EF predictions
    F_a  = 0.5 * (A_hat * g * omega / (N_0**2))**3 * (k_z**2 / (k_x**2) + 1)
    F_p  = -(A_hat * g * omega / (N_0**2))**2 * (omega * k_z / (k_x**2))
    F_nu = -nu * k_z * (A_hat * g * omega / (N_0**2))**2 * (k_z**2 / (k_x**2) + 1)
    print('\tPredictions for N_0 = {0}, omega = {1}, k = {2}'.format(N_0, omega, k))
    print('\ttheta = {0}, k_x = {1}, k_z = {2}'.format(np.rad2deg(theta), k_x, k_z))
    print('\tc_gz = {0}, n_T = {1}'.format(c_gz, n_T))
    print('\tF_a = {0}, F_p = {1}, F_nu = {2}'.format(F_a, F_p, F_nu))

# Make the predictions
print('Constant stratification predictions:')
make_ef_predictions(const_N_params)
print('Single layer stratification predictions:')
make_ef_predictions(single_layer_params)
print('Double layer stratification predictions:')
make_ef_predictions(double_layer_params)