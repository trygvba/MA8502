# Script for testing the convective term:

import sys
sys.path.insert(0, '../Mesh')
sys.path.insert(0, '../Laplace')

import time

# Numpy and Scipy:
import numpy as np
import scipy.linalg as la

# Own modules:
import laplace_functions as lp
import structured_grids as sg
import quadrature_nodes as qn
import gordon_hall as gh
import convection_functions as cf

# For plotting:
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

#####################################
# Boundary corners:
#####################################

x_corners = np.array( [ [0., 1.], [0., 1.] ] )
y_corners = np.array( [ [0., 0.], [1., 1.] ] )
#####################################
#   LOAD MAPPING:
####################################
mu = 1.
def loadfunc1(x,y):
    return -x

def loadfunc2(x,y):
    return -y


######################################
# Number of GLL-points:
N = 40

print "N: ", N
print "Getting GLL-points and weights..."
xis = qn.GLL_points(N)
weights = qn.GLL_weights(N, xis)

# Coordinate matrices:
print "Using Gordon Hall to get coordinate matrices..."
X, Y = gh.gordon_hall_straight_line(0,0,x_corners, y_corners, xis, N)

# Getting lagrangian derivative matrix:
print "Getting Lagrangian derivative matrix."
D = sg.diff_matrix(xis, N)

# Get several derivative matrices:
X_xi = np.dot(X,D)
X_eta = np.dot(X.T,D)
Y_xi = np.dot(Y,D)
Y_eta = np.dot(Y.T,D)

# Get jacobian and total G matrix:
"Getting Jacobian and total 'G-matrix'..."
Jac, G_tot = lp.assemble_total_G_matrix(X_xi,
                                        X_eta,
                                        Y_xi,
                                        Y_eta,
                                        N,
                                        N)
# Assemble stiffness matrix:
tot_points = N**2
print "Assembling stiffness matrix..."
t1 = time.time()
A = lp.assemble_local_stiffness_matrix(D, G_tot, N, weights)
print "Stiffness assembly: ", time.time()-t1
# Assembling loading vector:

print "Assembling loading vector..."
t1 = time.time()
F1 = lp.assemble_loading_vector(X, Y, loadfunc1, Jac, weights)
F2 = lp.assemble_loading_vector(X, Y, loadfunc2, Jac, weights)
print "Loading assembly: ", time.time() - t1



###############################################
#       BOUNDARY CONDITIONS:
###############################################

print "Setting up boundary conditions..."
S = C + mu*A

# Lower side:
for I in range(N):
    S[I] = 0.
    S[I,I] = 1.
    F[I] = 0.

    # Upper side:
    ind = N*(N-1)+I
    S[ind] = 0.
    S[ind, ind] = 1.
    F[ind] = 0.


    # Left:
    indl = I*N
    S[indl] = 0.
    S[indl, indl] = 1.
    F[indl] = 0.


    # Right:
    indr = I*N+N-1
    S[indr] = 0.
    S[indr, indr] = 1.
    F[indr] = 0.


############################
#       SOLVING:
############################
print "Solving system...."
u = la.solve(S,F)

print "Done..."
u_exact = 0.5*(1-X**2)*(1-Y**2)



############################
#   PLOTTING:
############################
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X,Y,u.reshape((N,N)))


plt.show()







