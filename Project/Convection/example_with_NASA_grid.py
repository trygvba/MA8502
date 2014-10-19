# Here's a script showing how to use all our modules with the NASA grid
# on an advection-diffusion equation.

import sys
sys.path.insert(0, '../Mesh')
sys.path.insert(0, '../Laplace')

import time

# Numpy and Scipy:
import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse

# Own modules:
import laplace_functions as lp
import structured_grids as sg
import quadrature_nodes as qn
import gordon_hall as gh
import convection_functions as cf

# For plotting:
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
#############################################
#       LET'S GET STARTED:
#############################################

# Preliminary functions:
def loadfunc(x,y):
    return -1.

def v1(x,y):
    return x

def v2(x,y):
    return y



# Getting element coordinate matrices:
X_el, Y_el = sg.read_PLOT3D('../Mesh/NASA_grids/grid_coarse.xyz')
jdim, idim = X_el.shape

# Number of elements in the x-direction:
Nx = idim-1
Ny = jdim -1
# Get number of patch elements:
patches = sg.get_number_of_patch_elements(X_el, Y_el, idim)

# Number of GLL-points:
N = 3
xis = qn.GLL_points(N)
weights = qn.GLL_weights(N,xis)

# Get derivative matrix:
D = sg.diff_matrix(xis,N)

# Get local-to-global mapping:
loc_glob = sg.local_to_global_top_down(idim, jdim,
                                        N, N,
                                        patch=True,
                                        num_patch=patches)


# Number of elements:
num_el = loc_glob.shape[0]

# Degrees of freedom:
dofs = np.max(loc_glob)+1
# Points per element:
el_points = N**2

print "Number of elements: ", num_el
print "Degress of freedom: ", dofs

###########################################
#       OFF TO THE ASSEMBLY PART:
###########################################

# Initialise Stiffness- and convection matrices:
A = np.zeros( (dofs, dofs) )
C = np.zeros( (dofs, dofs) )

# Initialise loading vector:
F = np.zeros( dofs )
#Initialse local coordinate matrices:
X = np.zeros( (num_el, N, N) )
Y = np.zeros( (num_el, N, N) )


# Initialise a v-vector:
v = np.zeros( 2*el_points)

# LET'S BEGIN ITERATING OVER EACH ELEMENT:
print "Starting assembly..."
t1 = time.time()
for K in range(num_el):
    indx = K%Nx
    indy = K/Nx

    #Get local coordinates:
    X[K], Y[K] = gh.gordon_hall_straight_line(indy, indx, X_el, Y_el, xis, N)
    
    X_xi = np.dot(X[K],D)
    X_eta = np.dot(X[K].T,D)
    Y_xi = np.dot(Y[K], D)
    Y_eta = np.dot(Y[K].T, D)

    # Get Jacobian and total G-matrix:
    Jac, G_tot = lp.assemble_total_G_matrix(X_xi,
                                            X_eta,
                                            Y_xi,
                                            Y_eta,
                                            N,
                                            N)

    # Insert contribution to stiffness matrix:
    A[np.ix_(loc_glob[K],loc_glob[K])] += lp.assemble_local_stiffness_matrix(D, G_tot, N, weights)

    # Insert contribution to convection matrix:
    v[:el_points] = v1(X[K],Y[K]).ravel()
    v[el_points:] = v2(X[K],Y[K]).ravel()
    C[np.ix_(loc_glob[K], loc_glob[K])] += cf.assemble_convection_matrix(v,
                                                                        X_xi,
                                                                        X_eta,
                                                                        Y_xi,
                                                                        Y_eta,
                                                                        D,
                                                                        N,
                                                                        weights)

    # Insert contribution to the loading vector:
    F[loc_glob[K]] += lp.assemble_loading_vector(X[K], Y[K], loadfunc, Jac, weights)

print "Assembly time: ", time.time()-t1


##########################################
#   BOUNDARY CONDITIONS:
##########################################
S = C + 3.*A

# Dirichlet on the airfoil:
# With m patch elements, the airfoil is described by the
# elements [m,...,Nx-1-m]
for i in range(patches,Nx-patches):
    # We want the bottom part of the elements:
    indices = loc_glob[i,:N]
    S[indices] = 0.
    S[indices,indices] = 1.
    F[indices] = 0.

# Dirichlet on outer part:
for i in range(Nx):
    # We want the top part of these elements:
    indices = loc_glob[Nx*(Ny-1)+i,N*(N-1):]
    S[indices] = 0.
    S[indices, indices] = 1.
    F[indices] = 0.

# Dirichlet on the rightmost part of the boundary:
for i in range(Ny):
    K = i*Nx
    indices = loc_glob[K, ::N]
    S[indices] = 0.
    S[indices, indices] = 1.
    F[indices] = 0.

    K = i*Nx+Nx-1
    tempind = N*np.arange(N)+N-1
    indices = loc_glob[K, tempind]
    S[indices] = 0.
    S[indices, indices] = 1.
    F[indices] = 0.


################################
#       SOLVING:
################################
print "Starting to solve..."
t1 = time.time()
U = la.solve(S,F)
print "Time to solve: ", time.time()-t1

################################
#       PLOTTING:
################################
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for K in range(num_el):
    ax.plot_wireframe(X[K],Y[K], U[loc_glob[K]].reshape( (N,N) ) )

plt.show()

    


