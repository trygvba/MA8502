# Script for testing the convective term:

import sys
sys.path.insert(0, '../Mesh')
sys.path.insert(0, '../Laplace')

import numpy as np
import scipy.linalg as la
import laplace_functions as lp
import structured_grids as sg
import quadrature_nodes as qn
import gordon_hall as gh
import convection_functions as cf

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

#####################################
# Boundary corners:
#####################################

x_corners = np.array( [ [-1., 1.], [-1., 1.] ] )
y_corners = np.array( [ [-1., -1.], [1., 1.] ] )
#####################################
#   LOAD MAPPING:
####################################
def loadfunc1(x,y):
    return 0.

def loadfunc2(x,y):
    return 1.


######################################
# Number of GLL-points:
N = 20
xis = qn.GLL_points(N)
weights = qn.GLL_weights(N, xis)

# Coordinate matrices:
X, Y = gh.gordon_hall_straight_line(0,0,x_corners, y_corners, xis, N)

# Getting lagrangian derivative matrix:
D = sg.diff_matrix(xis, N)

# Get several derivative matrices:
X_xi = np.dot(X,D)
X_eta = np.dot(X.T,D)
Y_xi = np.dot(Y,D)
Y_eta = np.dot(Y.T,D)

# Get jacobian and total G matrix:
Jac, G_tot = lp.assemble_total_G_matrix(X_xi,
                                        X_eta,
                                        Y_xi,
                                        Y_eta,
                                        N,
                                        N)
# Assemble stiffness matrix:
tot_points = N**2
A = np.zeros( (2*tot_points, 2*tot_points) )
A[:tot_points, :tot_points] = lp.assemble_local_stiffness_matrix(D, G_tot, N, weights)
A[tot_points:, tot_points:] = A[:tot_points, :tot_points]

# Assembling loading vector:
F = np.zeros( 2*tot_points )
F[:tot_points] = lp.assemble_loading_vector(X, Y, loadfunc1, Jac, weights)
F[:tot_points] = lp.assemble_loading_vector(X, Y, loadfunc2, Jac, weights)


#############################################
# Now for the shit with the vector field:
#############################################

# Vector field should be given by the Hermitian function H=0.5*(1-x**2)*(1-y**2)
# Then  v1 = H_y = -y*(1-x**2),
#       v2 = -H_x = x*(1-y**2).

v = np.zeros( 2*tot_points )
def v1(x,y):
    return -y*(1.-x**2)

def v2(x,y):
    return x*(1.-y**2)
#############################
v[:tot_points] = v1(X,Y).ravel()
v[tot_points:] = v2(X,Y).ravel()

C = cf.assemble_convection_matrix(v, X_xi, X_eta, Y_xi, Y_eta, D, N, weights)

###############################################
#       BOUNDARY CONDITIONS:
###############################################

S = C+A

# Lower side:
for I in range(N):
    S[I] = 0.
    S[I,I] = 1.
    F[I] = 0.
    S[I+tot_points] = 0.
    S[I+tot_points,I+tot_points] = 1.
    F[I+tot_points] = 1.

    # Upper side:
    ind = N*(N-1)+I
    S[ind] = 0.
    S[ind, ind] = 1.
    F[ind] = 0.

    S[ind+tot_points] = 0.
    S[ind+tot_points, ind+tot_points] = 1.
    F[ind+tot_points] = 0.

    # Left:
    indl = I*N
    S[indl] = 0.
    S[indl, indl] = 1.
    F[indl] = 0.

    S[indl+tot_points] = 0.
    S[indl+tot_points, indl+tot_points] = 1.
    F[indl+tot_points] = 0.

    # Right:
    indr = I*N+N-1
    S[indr] = 0.
    S[indr, indr] = 1.
    F[indr] = 0.

    S[indr+tot_points] = 0.
    S[indr+tot_points, indr+ tot_points] = 1.
    F[indr+tot_points] = 0.

############################
#       SOLVING:
############################
u = la.solve(S,F)


############################
#   PLOTTING:
############################
plt.quiver(X,Y, u[:tot_points].reshape((N,N)), u[tot_points:].reshape((N,N)))
plt.show()







