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


# For plotting:
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
#############################################
#       LET'S GET STARTED:
#############################################

# Preliminary functions:
def loadfunc(x,y):
    return 1.

def v1(x,y):
    return 0.

v1 = np.vectorize(v1)

def v2(x,y):
    return 0.

v2 = np.vectorize(v2)

##############################################
# GENERATE MESH
##############################################

##mesh inputs
thetadef = 30 #angle between elements 3,4 etc
num_el = 6

#constants
mu = 0.5
alpha = np.pi/10
v = 100
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
N_tot = N**2
xis = qn.GLL_points(N)
weights = qn.GLL_weights(N, xis)
xis_p = xis[1:-1]
weights_p = weights[1:-1]

X, Y = np.meshgrid(xis,xis)
X_p, Y_p = np.meshgrid(xis_p,xis_p)
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
t1 = time.time()
print "Assembling stiffness matrix."
t2= time.time()
A1 = lp.assemble_local_stiffness_matrix(D, G_tot1, N, weights)
A2 = lp.assemble_local_stiffness_matrix(D, G_tot2, N, weights)
A3 = lp.assemble_local_stiffness_matrix(D, G_tot3, N, weights)
A4 = lp.assemble_local_stiffness_matrix(D, G_tot4, N, weights)
A5 = lp.assemble_local_stiffness_matrix(D, G_tot5, N, weights)
A6 = lp.assemble_local_stiffness_matrix(D, G_tot6, N, weights)
A = np.zeros( (dofs, dofs) )
A[np.ix_(loc_glob[0], loc_glob[0]) ] += A1
A[np.ix_(loc_glob[1], loc_glob[1]) ] += A2
A[np.ix_(loc_glob[2], loc_glob[2]) ] += A3
A[np.ix_(loc_glob[3], loc_glob[3]) ] += A4
A[np.ix_(loc_glob[4], loc_glob[4]) ] += A5
A[np.ix_(loc_glob[5], loc_glob[5]) ] += A6
print "Time to make stiffness matrix", time.time()-t2
#need U1,U2 for each submatrix
U1_1,U2_1 = U1,U2
U1_2,U2_2 = U1,U2
U1_3,U2_3 = U1,U2
U1_4,U2_4 = U1,U2
U1_5,U2_5 = U1,U2
U1_6,U2_6 = U1,U2
t2 = time.time()
Cc1_1,Cc2_1= cf.assemble_const_convection_matrix(np.dot(X1,D), np.dot(X1.T,D), np.dot(Y1, D), np.dot(Y1.T, D), D, N, weights) #_1
Cc1_2,Cc2_2= cf.assemble_const_convection_matrix(np.dot(X2,D), np.dot(X2.T,D), np.dot(Y2, D), np.dot(Y2.T, D), D, N, weights) #_1
Cc1_3,Cc2_3= cf.assemble_const_convection_matrix(np.dot(X3,D), np.dot(X3.T,D), np.dot(Y3, D), np.dot(Y3.T, D), D, N, weights) #_1
Cc1_4,Cc2_4= cf.assemble_const_convection_matrix(np.dot(X4,D), np.dot(X4.T,D), np.dot(Y4, D), np.dot(Y4.T, D), D, N, weights) #_1
Cc1_5,Cc2_5= cf.assemble_const_convection_matrix(np.dot(X5,D), np.dot(X5.T,D), np.dot(Y5, D), np.dot(Y5.T, D), D, N, weights) #_1
Cc1_6,Cc2_6= cf.assemble_const_convection_matrix(np.dot(X6,D), np.dot(X6.T,D), np.dot(Y6, D), np.dot(Y6.T, D), D, N, weights) #_1

Cc1 = np.zeros( (dofs, dofs) ) #total C-matrix x-dir
Cc2 = np.zeros( (dofs, dofs) )

Cc1[np.ix_(loc_glob[0], loc_glob[0])] += Cc1_1
Cc2[np.ix_(loc_glob[0], loc_glob[0])] += Cc2_1
Cc1[np.ix_(loc_glob[1], loc_glob[1])] += Cc1_2
Cc2[np.ix_(loc_glob[1], loc_glob[1])] += Cc2_2
Cc1[np.ix_(loc_glob[2], loc_glob[2])] += Cc1_3
Cc2[np.ix_(loc_glob[2], loc_glob[2])] += Cc2_3
Cc1[np.ix_(loc_glob[3], loc_glob[3])] += Cc1_4
Cc2[np.ix_(loc_glob[3], loc_glob[3])] += Cc2_4
Cc1[np.ix_(loc_glob[4], loc_glob[4])] += Cc1_5
Cc2[np.ix_(loc_glob[4], loc_glob[4])] += Cc2_5
Cc1[np.ix_(loc_glob[5], loc_glob[5])] += Cc1_6
Cc2[np.ix_(loc_glob[5], loc_glob[5])] += Cc2_6

# UPDATING #########################
C1_1,C2_1 = cf.update_convection_matrix(U1_1,U2_1,Cc1_1,Cc2_1,N_tot)
C1_2,C2_2 = cf.update_convection_matrix(U1_2,U2_2,Cc1_2,Cc2_2,N_tot)
C1_3,C2_3 = cf.update_convection_matrix(U1_3,U2_3,Cc1_3,Cc2_3,N_tot)
C1_4,C2_4 = cf.update_convection_matrix(U1_4,U2_4,Cc1_4,Cc2_4,N_tot)
C1_5,C2_5 = cf.update_convection_matrix(U1_5,U2_5,Cc1_5,Cc2_5,N_tot)
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

print "Time to make convectino matrices", time.time()-t2


# Assemble local divergence matrix
print "Assembling divergence matrix."
B_loc = np.zeros((num_el,(N-2)**2,2*N**2))
B_loc = np.zeros((num_el,2*N**2,(N-2)**2))
t2 = time.time()
B_loc[0] = df.assemble_local_divergence_matrix(np.dot(X1,D), np.dot(X1.T,D), np.dot(Y1, D), np.dot(Y1.T, D), P_evals, D, weights, N)
B_loc[1] = df.assemble_local_divergence_matrix(np.dot(X2,D), np.dot(X2.T,D), np.dot(Y2, D), np.dot(Y2.T, D), P_evals, D, weights, N)
B_loc[2] = df.assemble_local_divergence_matrix(np.dot(X3,D), np.dot(X3.T,D), np.dot(Y3, D), np.dot(Y3.T, D), P_evals, D, weights, N)
B_loc[3] = df.assemble_local_divergence_matrix(np.dot(X4,D), np.dot(X4.T,D), np.dot(Y4, D), np.dot(Y4.T, D), P_evals, D, weights, N)
B_loc[4] = df.assemble_local_divergence_matrix(np.dot(X5,D), np.dot(X5.T,D), np.dot(Y5, D), np.dot(Y5.T, D), P_evals, D, weights, N)
B_loc[5] = df.assemble_local_divergence_matrix(np.dot(X6,D), np.dot(X6.T,D), np.dot(Y6, D), np.dot(Y6.T, D), P_evals, D, weights, N)

# Get local-to-global mapping:
B = np.zeros((2*dofs,num_el*(N-2)**2))
for i in range(num_el):
    dofx = loc_glob[i]
    dofy = loc_glob[i] + dofs
    dofp = loc_glob_p[i]
    B[np.ix_(np.hstack((dofx,dofy)),dofp)] += B_loc[i]
print "Time to make divergence matrices: ", time.time()-t2

#######################################################################################
#######################################################################################
#######################################################################################
############## MAKING THINGS GLOBALLY!!! ##############################################
#######################################################################################
#######################################################################################
#######################################################################################



# Defining S - Matrix
S1 = C1 + mu*A
S2 = C2 + mu*A
S = la.block_diag(S1,S2)
W = np.bmat([[S , -B],
            [B.T, np.zeros(shape=(B.shape[1],B.shape[1]))]])


F = np.zeros(num_el*(N-2)**2 + 2*dofs)

print "Assembly time: ", time.time()-t1, ", nice job, get yourself a beer."


#Imposing airfoil boundary:
for i in range(1,5):
    #Lower side of each element:
    indices = loc_glob[i,:N]
    indices = np.hstack( (indices, indices+dofs) )
    F[indices] = 0.
    W[indices] = 0.
    W[indices, indices] = 1.

# Imposing inflow:
for i in range(2,4):
    # Upper side of each element:
    indices = loc_glob[i,(N-1)*N:]
    
    # For x-direction:
    F[indices] = v*np.cos(alpha)
    W[indices] = 0.
    W[indices, indices] = 1.

    # For y-direction:
    indices =indices+dofs
    F[indices] = v*np.sin(alpha)
    W[indices] = 0.
    W[indices, indices] = 1.

m = N-3
W[2*dofs+m,:] = 0.
W[2*dofs+m,2*dofs+m] = 1.
F[2*dofs+m] = 0.


################################
#       SOLVING:
################################
eps = 1e-8
error = 1
counter = 1
N_it = 5
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
        indices = np.hstack( (indices, indices+dofs) )
        W[indices] = 0.
        W[indices, indices] = 1.

# Imposing inflow:
    for i in range(2,4):
        # Upper side of each element:
        indices = loc_glob[i,(N-1)*N:]
        
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

X[np.ix_(loc_glob[0], loc_glob[0]) ] =X1.ravel()
X[np.ix_(loc_glob[1], loc_glob[1]) ] =X2.ravel()
X[np.ix_(loc_glob[2], loc_glob[2]) ] =X3.ravel()
X[np.ix_(loc_glob[3], loc_glob[3]) ] =X4.ravel()
X[np.ix_(loc_glob[4], loc_glob[4]) ] =X5.ravel()
X[np.ix_(loc_glob[5], loc_glob[5]) ] =X6.ravel()

Y[np.ix_(loc_glob[0], loc_glob[0]) ] =Y1.ravel()
Y[np.ix_(loc_glob[1], loc_glob[1]) ] =Y2.ravel()
Y[np.ix_(loc_glob[2], loc_glob[2]) ] =Y3.ravel()
Y[np.ix_(loc_glob[3], loc_glob[3]) ] =Y4.ravel()
Y[np.ix_(loc_glob[4], loc_glob[4]) ] =Y5.ravel()
Y[np.ix_(loc_glob[5], loc_glob[5]) ] =Y6.ravel()

# QUIVERPLOT #
pl.quiver(X, Y, U1, U2, pivot='middle', scale=10, units='x')
pl.xlabel('$x$')
pl.ylabel('$y$')
pl.axis('image')
pl.show()
