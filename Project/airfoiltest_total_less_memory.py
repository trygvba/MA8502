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

# Preliminary functions:
def loadfunc(x,y):
    return 0.

def v1(x,y):
    return 0.

v1 = np.vectorize(v1)

def v2(x,y):
    return 0.

v2 = np.vectorize(v2)



# Order of GLL-points:
N = 10
num_el = 6 # Number of elements
N_it = 2
eps = 1e-8
xis = qn.GLL_points(N)
weights = qn.GLL_weights(N, xis)
xis_p = xis[1:-1]
weights_p = weights[1:-1]

#Local to global matrix:
loc_glob = sg.local_to_global_top_down(7, 2, N, N, patch=True, num_patch=1)
loc_glob_p = sg.local_to_global_pressure(7, 2, N, N)
dofs = np.max(loc_glob)+1

# Dimensions of resulting matrix: (not really sure how these go)
ydim = N
xdim = N

#Total number of points:
tot_points = N*N

# Initialise coordinate matrices:
X = np.zeros( tot_points )
Y = np.zeros( tot_points )


t1 = time.time()

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

for i in range(num_el):
    print "Assembling first Element..."
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

#Making global Convection matrix
C1,C2 = cf.update_convection_matrix(U1,U2,Cc1,Cc2,N)


# Defining S - Matrix
S = la.block_diag(S1,S2)
W = np.bmat([[S , -B],
            [B.T, np.zeros(shape=(B.shape[1],B.shape[1]))]])


F = np.zeros(num_el*(N-2)**2 + 2*dofs)

print "Assembly time: ", time.time()-t1, ", nice job, get yourself a beer."


#Imposing airfoil boundary:
for i in range(1,5):
    #Lower side of each element:
    indices = loc_glob[i,:N]
    #indices = loc_glob[i,(N-1)*N:]
    indices = np.hstack( (indices, indices+dofs) )
    F[indices] = 0.
    W[indices] = 0.
    W[indices, indices] = 1.

# Imposing inflow:
for i in range(2,4): #was 2,4
    # Upper side of each element:
    indices = loc_glob[i,-N:]
    #indices = loc_glob[i,:N]
    # For x-direction:
    F[indices] = v*np.cos(alpha)
    W[indices] = 0.
    W[indices, indices] = 1.

    # For y-direction:
    indices =indices+dofs
    F[indices] = v*np.sin(alpha)
    W[indices] = 0.
    W[indices, indices] = 1.

m = 0
W[2*dofs+m,:] = 0.
W[2*dofs+m,2*dofs+m] = 1.
F[2*dofs+m] = 0.


################################
#       SOLVING:
################################

error = 1.
counter = 1
UVP = la.solve(W,F)
U1 = UVP[:dofs]
U2 = UVP[dofs:2*dofs]

print "Time to solve              ", time.time()-t1
#U1 = la.solve(S1, F1)
#U2 = la.solve(S2, F2)

while (error>eps and counter <= N_it):
    t1 = time.time()
    print "Solving for the", counter ,"th time" 
    C1,C2 = cf.update_convection_matrix(U1,U2,Cc1,Cc2,dofs)
    S1 = C1 + mu*A
    S2 = C2 + mu*A
    S = la.block_diag(S1,S2)
    W = np.bmat([[S , -B],
                [B.T, np.zeros(shape=(B.shape[1],B.shape[1]))]])

#Imposing airfoil boundary:
    for i in range(1,5):
        #Lower side of each element:
        indices = loc_glob[i,:N]
        #indices = loc_glob[i,(N-1)*N:]
        indices = np.hstack( (indices, indices+dofs) )
        W[indices] = 0.
        W[indices, indices] = 1.

# Imposing inflow:
    for i in range(2,4):
        # Upper side of each element:
        indices = loc_glob[i,-N:]
        #indices = loc_glob[i,:N]
        # For x-direction:
        W[indices] = 0.
        W[indices, indices] = 1.

        # For y-direction:
        indices =indices+dofs
        W[indices] = 0.
        W[indices, indices] = 1.
    m = N-3
    W[2*dofs+m,:] = 0.
    W[2*dofs+m,2*dofs+m] = 1.
    F[2*dofs+m] = 0.

    print "Time to update", time.time()-t1
    print "Starting to solve..."
    t1 = time.time()
    UVP_new = la.solve(W,F)
    print "Time to solve: ", time.time()-t1
    error = float(la.norm(UVP_new - UVP))/la.norm(UVP)
    UVP = UVP_new
    U1 = UVP_new[:dofs]
    U2 = UVP_new[dofs:2*dofs]
    counter += 1
    print "Time to solve:             ", time.time()-t1
    print "error :                    ", error

################################
#       PLOTTING:
################################
# Making global X and Y coordinates
X = np.zeros((dofs,dofs))
Y = np.zeros((dofs,dofs))


fig = plt.figure(1)
# QUIVERPLOT #
plt.subplot(121)
pl.quiver(X1, Y1, U1[loc_glob[0]].reshape( (N,N) ), U2[loc_glob[0]].reshape( (N,N) ))
pl.quiver(X2, Y2, U1[loc_glob[1]].reshape( (N,N) ), U2[loc_glob[1]].reshape( (N,N) ))
pl.quiver(X3, Y3, U1[loc_glob[2]].reshape( (N,N) ), U2[loc_glob[2]].reshape( (N,N) )) 
pl.quiver(X4, Y4, U1[loc_glob[3]].reshape( (N,N) ), U2[loc_glob[3]].reshape( (N,N) ))
pl.quiver(X5, Y5, U1[loc_glob[4]].reshape( (N,N) ), U2[loc_glob[4]].reshape( (N,N) ))
pl.quiver(X6, Y6, U1[loc_glob[5]].reshape( (N,N) ), U2[loc_glob[5]].reshape( (N,N) ))
pl.xlabel('$x$')
pl.ylabel('$y$')
pl.axis('image')


# PRESSURE PLOT:
P = UVP[2*dofs:]
plt.subplot(122)
ax = fig.add_subplot(122, projection='3d')
ax.plot_wireframe(X1[1:-1,1:-1],Y1[1:-1,1:-1], P[loc_glob_p[0]].reshape( (N-2,N-2)))
ax.plot_wireframe(X2[1:-1,1:-1],Y2[1:-1,1:-1], P[loc_glob_p[1]].reshape( (N-2,N-2)))
ax.plot_wireframe(X3[1:-1,1:-1],Y3[1:-1,1:-1], P[loc_glob_p[2]].reshape( (N-2,N-2)))
ax.plot_wireframe(X4[1:-1,1:-1],Y4[1:-1,1:-1], P[loc_glob_p[3]].reshape( (N-2,N-2)))
ax.plot_wireframe(X5[1:-1,1:-1],Y5[1:-1,1:-1], P[loc_glob_p[4]].reshape( (N-2,N-2)))
ax.plot_wireframe(X6[1:-1,1:-1],Y6[1:-1,1:-1], P[loc_glob_p[5]].reshape( (N-2,N-2)))
pl.show()
