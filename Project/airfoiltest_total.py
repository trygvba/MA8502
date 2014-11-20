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
    return 1.

def v1(x,y):
    return 1.

v1 = np.vectorize(v1)

def v2(x,y):
    return 0.

v2 = np.vectorize(v2)

##############################################
# GENERATE MESH
##############################################

##mesh inputs
thetadef = 30 #angle between elements 3,4 etc

#constants
R = 507.79956092981
yrekt = 493.522687570593
xmax = 501.000007802345
xmin = -484.456616747519
dx = 0.001
xf = 0.2971739920760934289144667697 #x-value where f(x) has its max
##angle from xf to define elements 3 and 4 (in radians)


thetadef = thetadef*np.pi/180

#Global functions:
def outer_circle(theta):
    return R*np.cos(theta)+x0, -R*np.sin(theta)

def airfoil_upper(x):
    return 0.594689181*(0.298222773*np.sqrt(x) - 0.127125232*x - 0.357907906*x**2 + 0.291984971*x**3 - 0.105174606*x**4)

def airfoil_lower(x):
    return -0.594689181*(0.298222773*np.sqrt(x) - 0.127125232*x - 0.357907906*x**2 + 0.291984971*x**3 - 0.105174606*x**4)

def theta(x,y):
    return np.arccos((x-x0)/np.sqrt((x-x0)**2+y**2))

#crunch
x0 = R+xmin #origin of semicircle shape
xc = x0 + np.sqrt(R**2-yrekt**2) #intersection point in x; associated with yrekt
si = np.sin(thetadef)*(x0-xf) + airfoil_lower(xf)*np.cos(thetadef) + np.sqrt( R**2 - np.cos(thetadef)**2 * (xf + x0)**2 + 2*airfoil_lower(xf)*np.sin(thetadef)*np.cos(thetadef)*(xf-x0)- airfoil_lower(xf)**2 * np.sin(thetadef)**2)
#siupper = np.sin(thetadef)*(x0-xf) + airfoil_upper(xf)*np.cos(thetadef) + np.sqrt( R**2 - np.cos(thetadef)**2 * (xf + x0)**2 + 2*airfoil_upper(xf)*np.sin(thetadef)*np.cos(thetadef)*(x0-xf)- airfoil_upper(xf)**2 * np.sin(thetadef)**2) #may not be necessary
xint = xf - si*np.sin(thetadef) #intersection point in x
yint = np.absolute(-airfoil_upper(xf) - si*np.cos(thetadef)) #intersection point in y

###############################
#   BOUNDARY MAPS FOR MESH
###############################

#First element: (DONE)
def gamma_11(eta):
    return (xc+1)/2 + eta*(xc-1)/2 , -yrekt/2*(1+eta)

def gamma_12(xi):
    return (xc+xmax)/2 + xi*(xc-xmax)/2 , -yrekt

def gamma_13(eta):
    return xmax, -yrekt/2*(1+eta)

def gamma_14(xi):
    return 0.5*(1+xmax+xi*(1-xmax)) , 0

#Second element: (DONE)
def gamma_21(eta): #altered
    return (xf+xint)/2 + eta*(xint-xf)/2 , 0.5*( (-yint-airfoil_lower(xf))*eta + -yint+airfoil_lower(xf) )

def gamma_22(xi):
    theta1 = theta(xc,-yrekt)
    theta2 = theta(xint,-yint)
    return outer_circle( (theta2-theta1)/2*xi + (theta2+theta1)/2)

def gamma_23(eta):
    return gamma_11(eta)

def gamma_24(xi):
    return (xf+1+(xf-1)*xi)/2 , airfoil_lower((xf+1+(xf-1)*xi)/2)

#Third element: (DONE)
def gamma_31(eta):
    return xmin/2*eta + xmin/2 , 0

def gamma_32(xi):
    theta2 = np.pi
    theta1 = theta(xint,-yint)
    return outer_circle( (theta2-theta1)/2*xi + (theta2+theta1)/2)

def gamma_33(eta):
    return gamma_21(eta)

def gamma_34(xi):
    return (xf-(xf)*xi)/2 , airfoil_lower((xf-(xf)*xi)/2)

#Fourth element: (DONE)
def gamma_41(eta):
    return (xf+xint)/2 + eta*(xint-xf)/2 , 0.5*( (yint-airfoil_upper(xf))*eta + yint+airfoil_upper(xf) )
 
def gamma_42(xi): #shit is fucked
    theta1 = 0+np.pi
    theta2 = 2*np.pi-theta(xint,yint)
    return outer_circle( (theta2-theta1)/2*xi + (theta2+theta1)/2)

def gamma_43(eta):
    return gamma_31(eta)

def gamma_44(xi):
    return (xf+(xf)*xi)/2 , airfoil_upper((xf+(xf)*xi)/2)

#Fifth element: (DONE)
def gamma_51(eta):
    return (xc+1)/2 + eta*(xc-1)/2 , yrekt/2*(1+eta)

def gamma_52(xi): #shit is fucked no more
    theta2 = theta(xc,yrekt)+np.pi
    theta1 = theta(xint,yint)+np.pi
    return outer_circle( (theta2-theta1)/2*xi + (theta2+theta1)/2)

def gamma_53(eta):
    return gamma_41(eta)

def gamma_54(xi):
    return (xf+1+(1-xf)*xi)/2 , airfoil_upper((xf+1+(1-xf)*xi)/2)

#Sixth element: (DONE)
def gamma_61(eta):
    return xmax, yrekt/2*(1+eta)

def gamma_62(xi):
     return (xc+xmax)/2 + xi*(xmax-xc)/2 , yrekt 

def gamma_63(eta):
    return gamma_51(eta)

def gamma_64(xi):
    return 0.5*(1+xmax+xi*(xmax-1)) , 0

###############################################


# Order of GLL-points:
N = 20
xis = qn.GLL_points(N)
weights = qn.GLL_weights(N, xis)
xis_p = xis[1:-1]
weights_p = weights[1:-1]

X, Y = np.meshgrid(xis,xis)
X_p, Y_p = np.meshgrid(xis_p,xis_p)
#Local to global matrix:
Local_to_global = sg.local_to_global_top_down(7, 2, N, N, patch=True, num_patch=1)
dofs = np.max(Local_to_global)+1

# Dimensions of resulting matrix: (not really sure how these go)
ydim = N
xdim = N

#Total number of points:
tot_points = N*N

# Initialise coordinate matrices:
X = np.zeros( tot_points )
Y = np.zeros( tot_points )


t1 = time.time()

#Generate mesh:
print "Getting Mesh..."
X1, Y1 = gh.gordon_hall_grid( gamma_11, gamma_12, gamma_13, gamma_14, xis, xis)
X2, Y2 = gh.gordon_hall_grid( gamma_21, gamma_22, gamma_23, gamma_24, xis, xis)
X3, Y3 = gh.gordon_hall_grid( gamma_31, gamma_32, gamma_33, gamma_34, xis, xis)
X4, Y4 = gh.gordon_hall_grid( gamma_41, gamma_42, gamma_43, gamma_44, xis, xis)
X5, Y5 = gh.gordon_hall_grid( gamma_51, gamma_52, gamma_53, gamma_54, xis, xis)
X6, Y6 = gh.gordon_hall_grid( gamma_61, gamma_62, gamma_63, gamma_64, xis, xis)
######################################################################################################################################
######################################################################################################################################
######################################################################################################################################
#################################################MESH GENERATED#######################################################################
######################################################################################################################################
######################################################################################################################################
######################################################################################################################################

#Reshape the coordinate vector:
# This doesn't do anything I think.
X1 = X1.reshape( (ydim, xdim) )
Y1 = Y1.reshape( (ydim, xdim) )
X2 = X2.reshape( (ydim, xdim) )
Y2 = Y2.reshape( (ydim, xdim) )
X3 = X3.reshape( (ydim, xdim) )
Y3 = Y3.reshape( (ydim, xdim) )
X4 = X4.reshape( (ydim, xdim) )
Y4 = Y4.reshape( (ydim, xdim) )
X5 = X5.reshape( (ydim, xdim) )
Y5 = Y5.reshape( (ydim, xdim) )
X6 = X6.reshape( (ydim, xdim) )
Y6 = Y6.reshape( (ydim, xdim) )

# Get lagrangian derivative matrix:
print "Getting derivative matrix..."
D = sg.diff_matrix(xis, N)

print "Getting Jacobian and G-matrices."
Jac1, G_tot1 = lp.assemble_total_G_matrix(np.dot(X1,D),
                                        np.dot(X1.T,D),
                                        np.dot(Y1,D),
                                        np.dot(Y1.T,D),
                                        N,
                                        N)
Jac2, G_tot2 = lp.assemble_total_G_matrix(np.dot(X2,D),
                                        np.dot(X2.T,D),
                                        np.dot(Y2,D),
                                        np.dot(Y2.T,D),
                                        N,
                                        N)
Jac3, G_tot3 = lp.assemble_total_G_matrix(np.dot(X3,D),
                                        np.dot(X3.T,D),
                                        np.dot(Y3,D),
                                        np.dot(Y3.T,D),
                                        N,
                                        N)
Jac4, G_tot4 = lp.assemble_total_G_matrix(np.dot(X4,D),
                                        np.dot(X4.T,D),
                                        np.dot(Y4,D),
                                        np.dot(Y4.T,D),
                                        N,
                                        N)
Jac5, G_tot5 = lp.assemble_total_G_matrix(np.dot(X5,D),
                                        np.dot(X5.T,D),
                                        np.dot(Y5,D),
                                        np.dot(Y5.T,D),
                                        N,
                                        N)
Jac6, G_tot6 = lp.assemble_total_G_matrix(np.dot(X6,D),
                                        np.dot(X6.T,D),
                                        np.dot(Y6,D),
                                        np.dot(Y6.T,D),
                                        N,
                                        N)
P_evals = df.pressure_basis_point_evaluations(xis, N, D)
#########################################################
##################### ASSEMBLY ##########################
#########################################################

# initial conditions
U1 = v1(X,Y).ravel()
U2 = v2(X,Y).ravel()

# Now we're closing in on some shit. Prepare to ASSEMBLE!
# Assemble stiffness matrix:
print "Assembling stiffness matrix."
A1 = lp.assemble_local_stiffness_matrix(D, G_tot1, N, weights)
A2 = lp.assemble_local_stiffness_matrix(D, G_tot2, N, weights)
A3 = lp.assemble_local_stiffness_matrix(D, G_tot3, N, weights)
A4 = lp.assemble_local_stiffness_matrix(D, G_tot4, N, weights)
A5 = lp.assemble_local_stiffness_matrix(D, G_tot5, N, weights)
A6 = lp.assemble_local_stiffness_matrix(D, G_tot6, N, weights)
A = np.zeros( (dofs, dofs) )
A[np.ix_(Local_to_global[0], Local_to_global[0]) ] += A1
A[np.ix_(Local_to_global[1], Local_to_global[1]) ] += A2
A[np.ix_(Local_to_global[2], Local_to_global[2]) ] += A3
A[np.ix_(Local_to_global[3], Local_to_global[3]) ] += A4
A[np.ix_(Local_to_global[4], Local_to_global[4]) ] += A5
A[np.ix_(Local_to_global[5], Local_to_global[5]) ] += A6
print "Time to make stiffness matrix", time.time()-t2

# Assemble constant convection matrix
print "Assembling convection matrices."
t1 = time.time()
C = np.zeros( (dofs, dofs) )
Cc1,Cc2= cf.assemble_const_convection_matrix(X_xi, X_eta, Y_xi, Y_eta, D, N, weights)
C1,C2 = cf.update_convection_matrix(U1,U2,Cc1,Cc2,N_tot)
print "Time to make convection matrices: ", time.time()-t1
# Get local-to-global mapping:
loc_glob = sg.local_to_global_top_down(7, 2, N, N, patch=True, num_patch=1)

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

# Initialise loading vector:
F = np.zeros( dofs )


# Initialise six v-vectors:
vec1 = np.zeros( 2*el_points)
vec2 = np.zeros( 2*el_points)
vec3 = np.zeros( 2*el_points)
vec4 = np.zeros( 2*el_points)
vec5 = np.zeros( 2*el_points)
vec6 = np.zeros( 2*el_points)

#ASSEMBLE LAPLACE OPERATOR 
#ALREADY ASSEMBLED, BITCH

# LET'S BEGIN ITERATING OVER EACH ELEMENT:
print "Starting assembly..."
t1 = time.time()
#for K in range(num_el):
#indx = K%Nx
#indy = K/Nx

    #Get local coordinates:
    #X[K], Y[K] = gh.gordon_hall_straight_line(indy, indx, X_el, Y_el, xis, N)
    
#X_xi = np.dot(X[K],D)
#X_eta = np.dot(X[K].T,D)
#Y_xi = np.dot(Y[K], D)
#Y_eta = np.dot(Y[K].T, D)


    # Insert contribution to stiffness matrix:
    #A[np.ix_(loc_glob[K],loc_glob[K])] += lp.assemble_local_stiffness_matrix(D, G_tot, N, weights)

    # Insert contribution to convection matrix:
vec1[:el_points] = v1(X1,Y1).ravel()
vec1[el_points:] = v2(X1,Y1).ravel()
vec2[:el_points] = v1(X2,Y2).ravel()
vec2[el_points:] = v2(X2,Y2).ravel()
vec3[:el_points] = v1(X3,Y3).ravel()
vec3[el_points:] = v2(X3,Y3).ravel()
vec4[:el_points] = v1(X4,Y4).ravel()
vec4[el_points:] = v2(X4,Y4).ravel()
vec5[:el_points] = v1(X5,Y5).ravel()
vec5[el_points:] = v2(X5,Y5).ravel()
vec6[:el_points] = v1(X6,Y6).ravel()
vec6[el_points:] = v2(X6,Y6).ravel()
#need U1,U2 for each submatrix
Cc1_1,Cc2_1= cf.assemble_const_convection_matrix(np.dot(X1,D), np.dot(X1.T,D), np.dot(Y1, D), np.dot(Y1.T, D), D, N, weights) #_1
C1_1,C2_1 = cf.update_convection_matrix(U1_1,U2_1,Cc1_1,Cc2_1,N_tot)
Cc1_2,Cc2_2= cf.assemble_const_convection_matrix(np.dot(X2,D), np.dot(X2.T,D), np.dot(Y2, D), np.dot(Y2.T, D), D, N, weights) #_1
C1_2,C2_2 = cf.update_convection_matrix(U1_2,U2_2,Cc1_2,Cc2_2,N_tot)
Cc1_3,Cc2_3= cf.assemble_const_convection_matrix(np.dot(X3,D), np.dot(X3.T,D), np.dot(Y3, D), np.dot(Y3.T, D), D, N, weights) #_1
C1_3,C2_3 = cf.update_convection_matrix(U1_3,U2_3,Cc1_3,Cc2_3,N_tot)
Cc1_4,Cc2_4= cf.assemble_const_convection_matrix(np.dot(X4,D), np.dot(X4.T,D), np.dot(Y4, D), np.dot(Y4.T, D), D, N, weights) #_1
C1_4,C2_4 = cf.update_convection_matrix(U1_4,U2_4,Cc1_4,Cc2_4,N_tot)
Cc1_5,Cc2_5= cf.assemble_const_convection_matrix(np.dot(X5,D), np.dot(X5.T,D), np.dot(Y5, D), np.dot(Y5.T, D), D, N, weights) #_1
C1_5,C2_5 = cf.update_convection_matrix(U1_5,U2_5,Cc1_5,Cc2_5,N_tot)
Cc1_6,Cc2_6= cf.assemble_const_convection_matrix(np.dot(X6,D), np.dot(X6.T,D), np.dot(Y6, D), np.dot(Y6.T, D), D, N, weights) #_1
C1_6,C2_6 = cf.update_convection_matrix(U1_6,U2_6,Cc1_6,Cc2_6,N_tot)
C1 = np.zeros( (dofs, dofs) ) #total C-matrix x-dir
C2 = np.zeros( (dofs, dofs) )
C1[np.ix_(loc_glob[0], loc_glob[0])] += C1_1
C2[np.ix_(loc_glob[0], loc_glob[0])] += C2_1
C1[np.ix_(loc_glob[1], loc_glob[1])] += C1_2
C2[np.ix_(loc_glob[1], loc_glob[1])] += C2_2
C1[np.ix_(loc_glob[2], loc_glob[2])] += C1_3
C2[np.ix_(loc_glob[2], loc_glob[2])] += C2_3
C1[np.ix_(loc_glob[3], loc_glob[3])] += C1_4
C2[np.ix_(loc_glob[3], loc_glob[3])] += C2_4
C1[np.ix_(loc_glob[4], loc_glob[4])] += C1_5
C2[np.ix_(loc_glob[4], loc_glob[4])] += C2_5
C1[np.ix_(loc_glob[5], loc_glob[5])] += C1_6
C2[np.ix_(loc_glob[5], loc_glob[5])] += C2_6


# Assemble local divergence matrix
print "Assembling divergence matrix."
t1 = time.time()
B1 = df.assemble_local_divergence_matrix(np.dot(X1,D), np.dot(X1.T,D), np.dot(Y1, D), np.dot(Y1.T, D), P_evals, D, weights, N)
B2 = df.assemble_local_divergence_matrix(np.dot(X2,D), np.dot(X2.T,D), np.dot(Y2, D), np.dot(Y2.T, D), P_evals, D, weights, N)
B3 = df.assemble_local_divergence_matrix(np.dot(X3,D), np.dot(X3.T,D), np.dot(Y3, D), np.dot(Y3.T, D), P_evals, D, weights, N)
B4 = df.assemble_local_divergence_matrix(np.dot(X4,D), np.dot(X4.T,D), np.dot(Y4, D), np.dot(Y4.T, D), P_evals, D, weights, N)
B5 = df.assemble_local_divergence_matrix(np.dot(X5,D), np.dot(X5.T,D), np.dot(Y5, D), np.dot(Y5.T, D), P_evals, D, weights, N)
B6 = df.assemble_local_divergence_matrix(np.dot(X6,D), np.dot(X6.T,D), np.dot(Y6, D), np.dot(Y6.T, D), P_evals, D, weights, N)
B = np.zeros( (dofs, dofs-2) )
B[np.ix_(loc_glob[0], loc_glob[0])] += B1
B[np.ix_(loc_glob[1], loc_glob[1])] += B2
B[np.ix_(loc_glob[2], loc_glob[2])] += B3
B[np.ix_(loc_glob[3], loc_glob[3])] += B4
B[np.ix_(loc_glob[4], loc_glob[4])] += B5
B[np.ix_(loc_glob[5], loc_glob[5])] += B6


print "Time to make divergence matrices: ", time.time()-t1

# Assemble loading vector:
print "Assembling loading vector."
F1 = lp.assemble_loading_vector(X, Y, f1, Jac, weights)
F2 = lp.assemble_loading_vector(X, Y, f2, Jac, weights)
F3 = lp.assemble_loading_vector(X_p, Y_p, f3, Jac, weights_p)



print "Assembling loading vector."
F1 = lp.assemble_loading_vector(X1, Y1, loadfunc, Jac1, weights)
F2 = lp.assemble_loading_vector(X2, Y2, loadfunc, Jac2, weights)
F3 = lp.assemble_loading_vector(X3, Y3, loadfunc, Jac3, weights)
F4 = lp.assemble_loading_vector(X4, Y4, loadfunc, Jac4, weights)
F5 = lp.assemble_loading_vector(X5, Y5, loadfunc, Jac5, weights)
F6 = lp.assemble_loading_vector(X6, Y6, loadfunc, Jac6, weights)

F[Local_to_global[0]] += F1
F[Local_to_global[1]] += F2
F[Local_to_global[2]] += F3
F[Local_to_global[3]] += F4
F[Local_to_global[4]] += F5
F[Local_to_global[5]] += F6
print "Assembly time: ", time.time()-t1, ", nice job, get yourself a beer."


##########################################
#   BOUNDARY CONDITIONS:
##########################################
S = C + mu*A

# Upper part:
for k in range(6):
    indices = Local_to_global[k,-N:]
    S[indices] = 0.
    F[indices] = 0.
    S[indices,indices] = 1.

# Lower part:
for k in range(1,5):
    indices = Local_to_global[k,:N]
    S[indices] = 0.
    F[indices] = 1.
    S[indices, indices] = 1.

#Left part of the first element:
indices = Local_to_global[0,::N]
S[indices] = 0.
F[indices] = 0.
S[indices,indices] = 1.

# Right part of the last element:
indices = Local_to_global[-1, N-1 + N*np.arange(N)]
S[indices] = 0.
F[indices] = 0.
S[indices, indices] = 1.

print "Now for the Direchlet stuff"

# Dirichlet on outer part: DO SOMETHING WITH THIS
#for i in range(Nx):
#    # We want the top part of these elements:
#    indices = loc_glob[Nx*(Ny-1)+i,N*(N-1):]
#    S[indices] = 0.
#    S[indices, indices] = 1.
#    F[indices] = 0.

# Dirichlet on the rightmost part of the boundary:
# ACTUALLY: Let's just keep homogeneous Neumann here,
# The discretisation of he wake is not suited for a
# boundary layer here.

#for i in range(Ny):
#    K = i*Nx
#    indices = loc_glob[K, ::N]
#    S[indices] = 0.
#    S[indices, indices] = 1.
#    F[indices] = 0.
#
#    K = i*Nx+Nx-1
#    tempind = N*np.arange(N)+N-1
#    indices = loc_glob[K, tempind]
#    S[indices] = 0.
#    S[indices, indices] = 1.
#    F[indices] = 0.


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
U_min, U_max = -np.abs(U).max(), np.abs(U).max()
#fig = plt.figure()
#ax = fig.add_subplot(111)

#for K in range(num_el):
#    temp = U[loc_glob[K]].reshape((N,N))
temp = U[loc_glob[0]].reshape((N,N))
plt.pcolormesh(X1,Y1, temp[:-1,:-1], cmap='RdBu', vmin=U_min, vmax=U_max)
temp = U[loc_glob[1]].reshape((N,N))
plt.pcolormesh(X2,Y2, temp[:-1,:-1], cmap='RdBu', vmin=U_min, vmax=U_max)
temp = U[loc_glob[2]].reshape((N,N))
plt.pcolormesh(X3,Y3, temp[:-1,:-1], cmap='RdBu', vmin=U_min, vmax=U_max)
temp = U[loc_glob[3]].reshape((N,N))
plt.pcolormesh(X4,Y4, temp[:-1,:-1], cmap='RdBu', vmin=U_min, vmax=U_max)
temp = U[loc_glob[4]].reshape((N,N))
plt.pcolormesh(X5,Y5, temp[:-1,:-1], cmap='RdBu', vmin=U_min, vmax=U_max)
temp = U[loc_glob[5]].reshape((N,N))
plt.pcolormesh(X6,Y6, temp[:-1,:-1], cmap='RdBu', vmin=U_min, vmax=U_max)

for i in range(N):
	plt.plot(X1[:,i], Y1[:,i],'b')
	plt.plot(X1[i,:], Y1[i,:], 'b')
	plt.plot(X2[:,i], Y2[:,i],'b')
	plt.plot(X2[i,:], Y2[i,:], 'b')
	plt.plot(X3[:,i], Y3[:,i],'b')
	plt.plot(X3[i,:], Y3[i,:], 'b')
	plt.plot(X4[:,i], Y4[:,i],'b')
	plt.plot(X4[i,:], Y4[i,:], 'b')
	plt.plot(X5[:,i], Y5[:,i],'b')
	plt.plot(X5[i,:], Y5[i,:], 'b')
	plt.plot(X6[:,i], Y6[:,i],'b')
	plt.plot(X6[i,:], Y6[i,:], 'b')

plt.show()


