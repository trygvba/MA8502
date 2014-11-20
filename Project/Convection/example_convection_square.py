# Script for testing all the terms:
# Testing on a square

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
import divergence_functions as df

# For plotting:
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.pylab as pl

# Defining loading function
def f1(x,y):
    return 0.

def f2(x,y):
    return 0.

def f3(x,y):
    return 0.

# Defining initial values
def v1(x,y):
    return 1.

v1 = np.vectorize(v1)

def v2(x,y):
    return 0.

v2 = np.vectorize(v2)

# Defining Diffusion constant
mu = 0.00001


#####################################
####### Generating mesh #############
N = 35
N_tot = N**2
xis = qn.GLL_points(N)
xis_p = xis[1:-1]
weights = qn.GLL_weights(N, xis)
weights_p = weights[1:-1]

X, Y = np.meshgrid(xis,xis)
X_p, Y_p = np.meshgrid(xis_p,xis_p)
#####################################

t1 = time.time()
# Get lagrangian derivative matrix:
print "Getting derivative matrix..."
D = sg.diff_matrix(xis, N)

X_xi = np.dot(X,D)
X_eta =np.dot(X.T,D)
Y_xi = np.dot(Y,D)
Y_eta =np.dot(Y.T,D)

# Get Jacobian and total G-matrix:
print "Getting Jacobian and G-matrices."
Jac, G_tot = lp.assemble_total_G_matrix(X_xi, X_eta, Y_xi, Y_eta,N,N)
  
P_evals = df.pressure_basis_point_evaluations(xis, N, D)
print "Time to make auxiliary matrices", time.time()-t1
#########################################################
##################### ASSEMBLY ##########################
#########################################################

# initial conditions
U1 = v1(X,Y).ravel()
U2 = v2(X,Y).ravel()

# Now we're closing in on some shit. Prepare to ASSEMBLE!
# Assemble stiffness matrix:
print "Assembling stiffness matrix."
t2 = time.time()
A = lp.assemble_local_stiffness_matrix(D, G_tot, N, weights)
print "Time to make stiffness matrix", time.time()-t2

# Assemble constant convection matrix
print "Assembling convection matrices."
t1 = time.time()
Cc1,Cc2= cf.assemble_const_convection_matrix(X_xi, X_eta, Y_xi, Y_eta, D, N, weights)
C1,C2 = cf.update_convection_matrix(U1,U2,Cc1,Cc2,N_tot)
print "Time to make convection matrices: ", time.time()-t1

# Assemble local divergence matrix
print "Assembling divergence matrix."
t1 = time.time()
B = df.assemble_local_divergence_matrix(X_xi, X_eta, Y_xi, Y_eta, P_evals, D, weights, N)
print "Time to make divergence matrices: ", time.time()-t1

# Assemble loading vector:
print "Assembling loading vector."
F1 = lp.assemble_loading_vector(X, Y, f1, Jac, weights)
F2 = lp.assemble_loading_vector(X, Y, f2, Jac, weights)
F3 = lp.assemble_loading_vector(X_p, Y_p, f3, Jac, weights_p)


# Defining S - Matrix
S1 = C1 + mu*A
S2 = C2 + mu*A

print "Time to assemble all blockmatrices: ", time.time()-t2

#########################################################
##################### ASSEMBLY - END ####################
#########################################################

print "Imposing boundary conditions."
# Now we need to handle the Dirichlet Boundary conditions:
for i in range(N):
    #Lower side:
    S1[i,:] = 0.
    S1[i,i] = 1.
    F1[i] = 0.

    #Upper side:
    S1[N*(N-1)+i,:] = 0.
    S1[N*(N-1)+i, N*(N-1)+i] = 1.
    F1[N*(N-1)+i] = 0.

    #Left side:
    S1[i*N,:] = 0.
    S1[i*N, i*N] = 1.
    F1[i*N] = 1.-Y[i,0]**2

    #Right side:
#   S1[i*N+N-1,:] = 0.
#   S1[i*N+N-1, i*N+N-1] =1.
#   F1[i*N+N-1] = F[i*N]

######## Y-direction #########
    #Lower side:
    S2[i,:] = 0.
    S2[i,i] = 1.
    F2[i] = 0.

    #Upper side:
    S2[N*(N-1)+i,:] = 0.
    S2[N*(N-1)+i, N*(N-1)+i] = 1.
    F2[N*(N-1)+i] = 0.

    #Left side:
#    S2[i*N,:] = 0.
#    S2[i*N, i*N] = 1.
#    F2[i*N] = 0.

    #Right side:
#   S2[i*N+N-1,:] = 0.
#   S2[i*N+N-1, i*N+N-1] =1.
#   F2[i*N+N-1] = 0.


########################
#   SOLVING ITERATIVELY
########################
t1 = time.time()
S = la.block_diag(S1,S2)
F = np.append(F1,np.append(F2,F3))
W = np.bmat([[S , -B],
            [B.T, np.zeros(shape=(B.shape[1],B.shape[1]))]])

m = N-3
W[2*N**2+m,:] = 0.
W[2*N**2+m,2*N**2+m] = 1.
F[2*N**2+m] = 0.

eps = 1e-8
error = 1
counter = 1
N_it = 30
UVP = la.solve(W,F)
U1 = UVP[:N**2]
U2 = UVP[N**2:2*N**2]

print "Time to solve              ", time.time()-t1
#U1 = la.solve(S1, F1)
#U2 = la.solve(S2, F2)

while (error>eps and counter <= N_it):
  t1 = time.time()
  print "Solving for the", counter ,"th time" 
  C1,C2 = cf.update_convection_matrix(U1,U2,Cc1,Cc2,N_tot)
  S1 = C1 + mu*A
  S2 = C2 + mu*A
# Now we need to handle the Dirichlet Boundary conditions:
  for i in range(N):
      # Conditions for the x-direction
      #Lower side:
      S1[i,:] = 0.
      S1[i,i] = 1.

      #Upper side:
      S1[N*(N-1)+i,:] = 0.
      S1[N*(N-1)+i, N*(N-1)+i] = 1.

      #Left side:
      S1[i*N,:] = 0.
      S1[i*N, i*N] = 1.

      #Right side:
#      S1[i*N+N-1,:] = 0.
#      S1[i*N+N-1, i*N+N-1] =1.

      # Conditions for the y-direction
      #Lower side:
      S2[i,:] = 0.
      S2[i,i] = 1.

      #Upper side:
      S2[N*(N-1)+i,:] = 0.
      S2[N*(N-1)+i, N*(N-1)+i] = 1.

      #Left side:
#      S2[i*N,:] = 0.
#      S2[i*N, i*N] = 1.

      #Right side:
#      S2[i*N+N-1,:] = 0.
#      S2[i*N+N-1, i*N+N-1] =1.
############ BC - END #################

  S = la.block_diag(S1,S2)
  W = np.bmat([[S , -B],
            [B.T, np.zeros(shape=(B.shape[1],B.shape[1]))]])
  W[2*N**2+m,:] = 0.
  W[2*N**2+m,2*N**2+m] = 1.
  print "Time to update total matrix", time.time()-t1
  t1 = time.time()
  UVP_new = la.solve(W, F)
#  error = float(la.norm(UVP_new - UVP))/la.norm(UVP)
  error = np.max(UVP_new - UVP)
  UVP = UVP_new
  U1 = UVP[:N**2]
  U2 = UVP[N**2:2*N**2]
  counter += 1
  print "Time to solve:             ", time.time()-t1
  print "error :                    ", error


#   TIME TO SOLVE:
####################
print "Done solving..."
P = UVP[2*N**2:]
P = np.reshape ( P, (N-2,N-2) )
U1 = np.reshape ( U1, (N,N) )
U2 = np.reshape ( U2, (N,N) )

t2 = time.time()-t1
print "Total time: ", t2

############################
# PLOTTING THE SHIT:
############################
# Quiverplot
fig = plt.figure(1)
plt.subplot(121)
pl.quiver(X, Y, U1, U2, pivot='middle', headwidth=4, headlength=6)
pl.xlabel('$x$')
pl.ylabel('$y$')
pl.axis('image')


# Pressureplot
plt.subplot(122)
ax = fig.add_subplot(122, projection='3d')
ax.plot_wireframe(X_p, Y_p, P)#-np.sin(2*np.pi*(X**2 + Y**2)))
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()
