# This is a script for testing the functions assembling the
# stiffness matrix and loading vector for the Poisson equation.
# The script may also serve as illustrative example.

import sys
sys.path.insert(0, '../Mesh')
import numpy as np
import scipy.linalg as la
import laplace_functions as lp
import structured_grids as sg
import quadrature_nodes as qn
import gordon_hall as gh

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
###########################
#   MAPPINGS:
###########################
#def gamma1(eta):
#    theta = np.pi/4.*(eta+1)
#    return 2.*np.cos(theta), 2.*np.sin(theta)

#def gamma2(xi):
#    return 0., 0.5*(3.+xi)

#def gamma3(eta):
#    theta = np.pi/4.*(eta+1)
#    return np.cos(theta), np.sin(theta)

#def gamma4(xi):
#    return 0.5*(3.+xi), 0.

#def f(x,y):
#    return 1.

#Boundary maps for a circle:
def gamma1(eta):
    theta = np.pi/4.*(eta+1.)
    return np.cos(theta), np.sin(theta)

def gamma2(xi):
    theta = np.pi/4.*(3.-xi)
    return np.cos(theta), np.sin(theta)

def gamma3(eta):
    theta = np.pi/4.*(5.-eta)
    return np.cos(theta), np.sin(theta)

def gamma4(xi):
    theta = np.pi/4.*(7.+xi)
    return np.cos(theta), np.sin(theta)

def f(x,y):
    r2 = x**2+y**2
    return -8.*np.pi*np.cos(2*np.pi*r2) + 16.*np.pi**2*r2*np.sin(2*np.pi*r2)

###########################
# Order of GLL-points:
N = 50
xis = qn.GLL_points(N)
weights = qn.GLL_weights(N, xis)

#Generate mesh:
print "Getting Mesh..."
X, Y = gh.gordon_hall_grid( gamma1, gamma2, gamma3, gamma4, xis, xis)

# Get lagrangian derivative matrix:
print "Getting derivative matrix..."
D = sg.diff_matrix(xis, N)

# Get Jacobian and total G-matrix:
print "Getting Jacobian and G-matrices."
Jac, G_tot = lp.assemble_total_G_matrix(np.dot(X,D),
                                        np.dot(X.T,D),
                                        np.dot(Y,D),
                                        np.dot(Y.T,D),
                                        N,
                                        N)
  
# Now we're closing in on some shit. Prepare to ASSEMBLE!
# Assemble stiffness matrix:
print "Assembling stiffness matrix."
A = lp.assemble_local_stiffness_matrix(D, G_tot, N, weights)

# Assemble loading vector:
print "Assembling loading vector."
F = lp.assemble_loading_vector(X, Y, f, Jac, weights)

print "Imposing boundary conditions."
# Now we need to handle the Dirichlet Boundary conditions:
for i in range(N):
    #Lower side:
    A[i,:] = 0.
    A[i,i] = 1.
    F[i] = 0.

    #Upper side:
    A[N*(N-1)+i,:] = 0.
    A[N*(N-1)+i, N*(N-1)+i] = 1.
    F[N*(N-1)+i] = 0.

    #Left side:
    A[i*N,:] = 0.
    A[i*N, i*N] = 1.
    F[i*N] = 0.

    #Right side:
    A[i*N+N-1,:] = 0.
    A[i*N+N-1, i*N+N-1] =1.
    F[i*N+N-1] = 0.

####################
#   TIME TO SOLVE:
####################
print "Solving..."
U = la.solve(A, F)
print "Done solving..."
U = np.reshape ( U, (N,N) )

############################
# PLOTTING THE SHIT:
############################
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, U)

plt.show()

