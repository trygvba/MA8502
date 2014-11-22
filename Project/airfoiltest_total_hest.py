# Here's a script showing how to use all our modules with the NASA grid
# on an advection-diffusion equation.

import sys
sys.path.insert(0, 'Mesh')
sys.path.insert(0, 'Laplace')
sys.path.insert(0, 'Convection')

import time

# Numpy and Scipy:
import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as sla

# Own modules:
import laplace_functions as lp
import structured_grids as sg
import quadrature_nodes as qn
import gordon_hall as gh
import convection_functions as cf
import divergence_functions as df
import gammas as gm


# For plotting:
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
#############################################
#       LET'S GET STARTED:
#############################################


# Order of GLL-points:
N = 20
mu = 300
v = 1
alpha = np.pi/20
num_el = 6 # Number of elements
N_it = 6
eps = 1e-8
# Reference element and weights for Velocity and pressure
xis = qn.GLL_points(N)
weights = qn.GLL_weights(N, xis)
xis_p , weights_p = xis[1:-1], weights[1:-1]

#Local to global matrix:
loc_glob = sg.local_to_global_top_down(7, 2, N, N, patch=True, num_patch=1)
loc_glob_p = sg.local_to_global_pressure(7, 2, N, N)
dofs = np.max(loc_glob)+1

# Dimensions of resulting matrix: (not really sure how these go)
ydim = N
xdim = N

# Initialise coordinate matrices:

t1 = time.time()

# Preliminary functions:
def loadfunc(x,y):
    return 0.

def v1(x,y):
    return 0.

v1 = np.vectorize(v1)

def v2(x,y):
    return 0.

v2 = np.vectorize(v2)
######################### ASSEMBLING AS WE WALK ################## 
############### WILL SAVE A LOT OF MEMORY AND CODELINES!!! #######

# Get lagrangian derivative matrix:
print "Getting derivative matrix..."
D = sg.diff_matrix(xis, N)
P_evals = df.pressure_basis_point_evaluations(xis, N, D)
# Init conditions
#U1 = v1(X,Y).ravel()
#U2 = v2(X,Y).ravel()
U1 = np.zeros(dofs)
U2 = np.zeros(dofs)

# Initialize global matrices
A = np.zeros( (dofs, dofs) )
Cc1 = np.zeros( (dofs, dofs) ) #total constant C-matrix x-dir
Cc2 = np.zeros( (dofs, dofs) )
C1 = np.zeros( (dofs, dofs) ) #total C-matrix x-dir
C2 = np.zeros( (dofs, dofs) )
B = np.zeros((2*dofs,num_el*(N-2)**2))  # Total B matrix
tass= time.time()
for i in range(num_el):
    t1 = time.time()
    print "Assembling the", i+1,"th Element..."
    X1, Y1 = gh.gordon_hall_grid( gm.gammas(i,1),gm.gammas(i,2),gm.gammas(i,3),gm.gammas(i,4), xis, xis)
    X1 = X1.reshape( (ydim, xdim) )
    Y1 = Y1.reshape( (ydim, xdim) )
    Jac1, G_tot1 = lp.assemble_total_G_matrix(np.dot(X1,D),
                                        np.dot(X1.T,D),
                                        np.dot(Y1,D),
                                        np.dot(Y1.T,D),
                                        N,
                                        N)
# Stiffness matrix
    A1 = lp.assemble_local_stiffness_matrix(D, G_tot1, N, weights)
    A[np.ix_(loc_glob[i], loc_glob[i]) ] += A1
# constant Convection matrix
    Cc1_1,Cc2_1= cf.assemble_const_convection_matrix(np.dot(X1,D), np.dot(X1.T,D), np.dot(Y1, D), np.dot(Y1.T, D), D, N, weights) #_1
    Cc1[np.ix_(loc_glob[i], loc_glob[i])] += Cc1_1
    Cc2[np.ix_(loc_glob[i], loc_glob[i])] += Cc2_1
# Divergence matrix
    B_loc = df.assemble_local_divergence_matrix(np.dot(X1,D), np.dot(X1.T,D), np.dot(Y1, D), np.dot(Y1.T, D), P_evals, D, weights, N)
    dofx = loc_glob[i]
    dofy = loc_glob[i] + dofs
    dofp = loc_glob_p[i]
    B[np.ix_(np.hstack((dofx,dofy)),dofp)] += B_loc
    print "time to assemble one element:", time.time()-t1

#Making global Convection matrix
C1,C2 = cf.update_convection_matrix(U1,U2,Cc1,Cc2)
F1 = np.zeros(dofs)
F2 = np.zeros(dofs)
F3 = np.zeros(num_el*(N-2)**2)

S1 = C1+mu*A
S2 = C2+mu*A
W = np.dot((B[:dofs].T),np.dot(la.inv(S1),B[:dofs])) 
+   np.dot((B[dofs:].T),np.dot(la.inv(S2),B[dofs:]))
m = 5
W[m] = 0.
W[m,m] = 1.

print "Assembly time: ", time.time()-tass, ", nice job, get yourself a beer."

print "Imposing boundary conditions and inflow..."
tinflow = time.time()
'''
#Imposing airfoil boundary:
for i in range(1,5):
    #Lower side of each element:
    indices = loc_glob[i,:N]
    F1[indices] = 0.
    F2[indices] = 0.

# Imposing inflow:
for i in range(2,4): #was 2,4
    # Upper side of each element:
    indices = loc_glob[i,-N:]
    # For x-direction:
    F1[indices] = v*np.cos(alpha)

    # For y-direction:
    F2[indices] = v*np.sin(alpha)
'''
#Setting the pressure in one point

print "Imposing time:", time.time()-tinflow

#G3 = -(np.dot(B[:dofs].T,la.solve(S1,F1))
#+np.dot(B[dofs:].T,la.solve(S2,F2)))
G3 = np.zeros(num_el*(N-2)**2)
G3[m] = 1

################################
#       SOLVING:
################################
t1 = time.time()
error = 1.
counter = 1
# New solving process
P = la.solve(W,G3)
G1 = (F1+np.dot(B[:dofs],P))
G2 = (F2+np.dot(B[dofs:],P))
#Imposing airfoil boundary:
for i in range(1,5):
    #Lower side of each element:
    indices = loc_glob[i,:N]
    S1[indices] = 0.
    S1[indices, indices] = 1.
    S2[indices] = 0.
    S2[indices, indices] = 1.
    G1[indices] = 0.
    G2[indices] = 0.

# Imposing inflow:
for i in range(2,4): #was 2,4
    # Upper side of each element:
    indices = loc_glob[i,-N:]
    # For x-direction:
    G1[indices] = v*np.cos(alpha)
    S1[indices] = 0.
    S1[indices, indices] = 1.

    # For y-direction:
    G2[indices] = v*np.sin(alpha)
    S2[indices] = 0.
    S2[indices, indices] = 1.


U1 = la.solve(S1,G1)
U2 = la.solve(S2,G2)

# solved

print "Time to solve              ", time.time()-t1

while (error>eps and counter <= N_it):
    t1 = time.time()
    print "Solving for the", counter ,"th time" 
    C1,C2 = cf.update_convection_matrix(U1,U2,Cc1,Cc2)
    S1 = C1+mu*A
    S2 = C2+mu*A
    W =   np.dot((B[:dofs].T),np.dot(la.inv(S1),B[:dofs]))
    +     np.dot((B[dofs:].T),np.dot(la.inv(S2),B[dofs:]))
    W[m] = 0.
    W[m,m] = 1.
    t1 = time.time()
# New solving process
    P_new = la.solve(W,G3)
    G1 = (F1+np.dot(B[:dofs],P_new))
    G2 = (F2+np.dot(B[dofs:],P_new))
    print "Time to update", time.time()-t1
    print "Starting to solve..."
#Imposing airfoil boundary:
    for i in range(1,5):
        #Lower side of each element:
        indices = loc_glob[i,:N]
        G1[indices] = 0.
        G2[indices] = 0.
        S1[indices] = 0.
        S1[indices, indices] = 1.
        S2[indices] = 0.
        S2[indices, indices] = 1.

# Imposing inflow:
    for i in range(2,4): #was 2,4
        # Upper side of each element:
        indices = loc_glob[i,-N:]
        # For x-direction:
        G1[indices] = v*np.cos(alpha)
        S1[indices] = 0.
        S1[indices, indices] = 1.

        # For y-direction:
        G2[indices] = v*np.sin(alpha)
        S2[indices] = 0.
        S2[indices, indices] = 1.


    U1_new = la.solve(S1,G1)
    U2_new = la.solve(S2,G2)

# solved

    print "Time to solve: ", time.time()-t1
    error = float(la.norm(P_new - P))/la.norm(P)
    P = np.copy(P_new)
    U1 = np.copy(U1_new)
    U2 = np.copy(U2_new)
    counter += 1
    print "error :                    ", error

################################
#       PLOTTING:
################################
# Making global X and Y coordinates


fig = plt.figure(1)
# QUIVERPLOT #
for i in range(num_el):
  X,Y = gh.gordon_hall_grid(gm.gammas(i,1),gm.gammas(i,2),gm.gammas(i,3),gm.gammas(i,4), xis, xis)
  plt.subplot(121)
  pl.quiver(X,Y, U1[loc_glob[i]].reshape( (N,N) ), U2[loc_glob[i]].reshape( (N,N) ))
  pl.xlabel('$x$')
  pl.ylabel('$y$')
  pl.axis('image')

# PRESSURE PLOT:
plt.subplot(122)
ax = fig.add_subplot(122, projection='3d')
for i in range(num_el):
  X,Y = gh.gordon_hall_grid(gm.gammas(i,1),gm.gammas(i,2),gm.gammas(i,3),gm.gammas(i,4), xis, xis)
  ax.plot_wireframe(X[1:-1,1:-1],Y[1:-1,1:-1], P[loc_glob_p[i]].reshape( (N-2,N-2)))

pl.show()
