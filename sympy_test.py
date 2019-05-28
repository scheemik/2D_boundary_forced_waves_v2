"""
A script for testing out sympy. I am hoping to use it to solve a system of
equations for the eigenvalues and eigenvectors.

"""
"""
!pip3 install --upgrade
!pip3 install sympy
"""
# %%
import numpy as np
import math
from sympy import *

# %%
x, t, z, nu = symbols('x t z nu')
y = Function('y')
dsolve(Eq(y(t).diff(t,t) - y(t), exp(t)))

# %%

# Define a matrix
A = Matrix([[0,1],[-2,-3]])
# Take the determinant
A.det()
# Put into reduced row echelon form
A.rref()
# Find the {eigenvalues: multiplicities, ...}
A.eigenvals()
# Find the [(eigenvalue, multiplicity, [eigenvector]), ...]
A.eigenvects()
# %%

a, b, c, d = symbols('a, b, c, d', real=True)
M1_sym = Matrix([[  a,     a,        0],
                 [  b,     0,        c],
                 [  0,     d,        0]])
M1_sym.det()

# %%

k_x, k_z, omega, rho, N, M_det, om2 = symbols('k_x, k_z, omega, rho, N, M_det, om2', real=True)
M = Matrix([[  0,     0,        0,      I*k_x/rho],
            [    0,     0,     -1,      I*k_z/rho],
            [    0,      N**2,     0,       0   ],
            [  I*k_x,     I*k_z,      0,          0   ]])
"""
M = Matrix([[  -I*omega,     0,        0,      I*k_x/rho],
            [    0,     -I*omega,     -1,      I*k_z/rho],
            [    0,      N**2,     -I*omega,       0   ],
            [  I*k_x,     I*k_z,      0,          0   ]])
"""
M_det = M.det()
M_det

M.eigenvals()
