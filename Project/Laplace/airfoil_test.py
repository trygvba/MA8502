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
import time as time

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

###############################
# PRELIMINARY DATA
###############################
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
#   BOUNDARY MAPS:
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

######################################
#       LOADING FUNCTION:
######################################
def loadfunc(x,y):
    return 0.


###########################
# Order of GLL-points:
N = 50
xis = qn.GLL_points(N)
weights = qn.GLL_weights(N, xis)

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

#Reshape the coordinate vector:
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

###############################THIS MARKS THE POINT WHERE I LITERALLY COULDN'T

# Get Jacobian and total G-matrix (This should be done for each element):
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
                                        
print "Assembling stiffness matrix."
A1 = lp.assemble_local_stiffness_matrix(D, G_tot1, N, weights)
A2 = lp.assemble_local_stiffness_matrix(D, G_tot2, N, weights)
A3 = lp.assemble_local_stiffness_matrix(D, G_tot3, N, weights)
A4 = lp.assemble_local_stiffness_matrix(D, G_tot4, N, weights)
A5 = lp.assemble_local_stiffness_matrix(D, G_tot5, N, weights)
A6 = lp.assemble_local_stiffness_matrix(D, G_tot6, N, weights)

###Now need a crafty Jew to assemble the global stiffness matrix A
#Using the Local_to_global mapping to insert at the right place:
A = np.zeros((dofs, dofs))#initialise the cumdumpster

###########################################line of fuckups beyond this line beyond this line beyond this line
#error reads: "array is not broadcastable to correct shape"
A[np.ix_(Local_to_global[0], Local_to_global[0]) ] += A1
A[np.ix_(Local_to_global[1], Local_to_global[1]) ] += A2
A[np.ix_(Local_to_global[2], Local_to_global[2]) ] += A3
A[np.ix_(Local_to_global[3], Local_to_global[3]) ] += A4
A[np.ix_(Local_to_global[4], Local_to_global[4]) ] += A5
A[np.ix_(Local_to_global[5], Local_to_global[5]) ] += A6
# Your Crafty Jew has arrived.

###
# Assemble loading vector:
print "Assembling loading vector."
F1 = lp.assemble_loading_vector(X1, Y1, loadfunc, Jac1, weights)
F2 = lp.assemble_loading_vector(X2, Y2, loadfunc, Jac2, weights)
F3 = lp.assemble_loading_vector(X3, Y3, loadfunc, Jac3, weights)
F4 = lp.assemble_loading_vector(X4, Y4, loadfunc, Jac4, weights)
F5 = lp.assemble_loading_vector(X5, Y5, loadfunc, Jac5, weights)
F6 = lp.assemble_loading_vector(X6, Y6, loadfunc, Jac6, weights)

###Now also need a crafty Jew to assemble the global loading vector F
#Same here:
F = np.zeros( dofs )
F[Local_to_global[0]] += F1
F[Local_to_global[1]] += F2
F[Local_to_global[2]] += F3
F[Local_to_global[3]] += F4
F[Local_to_global[4]] += F5
F[Local_to_global[5]] += F6

#########################################
#       BOUNDARY CONDITIONS:
#########################################

# I will try having zero Dirichlet on the outer boundary
# and 1 at the airfoil.

# This corresponds to the upper part of each element, and lower
# part of all but the first and last element. In addition the left
# part of the first, and the right part of the last must be handled.

# Upper part:
for k in range(6):
    indices = Local_to_global[k,-N:]
    A[indices] = 0.
    F[indices] = 0.
    A[indices,indices] = 1.

# Lower part:
for k in range(1,5):
    indices = Local_to_global[k,:N]
    A[indices] = 0.
    F[indices] = 1.
    A[indices, indices] = 1.

#Left part of the first element:
indices = Local_to_global[0,::N]
A[indices] = 0.
F[indices] = 0.
A[indices,indices] = 1.

# Right part of the last element:
indices = Local_to_global[-1, N-1 + N*np.arange(N)]
A[indices] = 0.
F[indices] = 0.
A[indices, indices] = 1.
#####################
#   SOLVE:
#####################
U = la.solve(A,F)

##########################################
#       TRYING TO PLOT:
##########################################
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_wireframe(X1,Y1, U[Local_to_global[0]].reshape( (N,N) ))
ax.plot_wireframe(X2,Y2, U[Local_to_global[1]].reshape( (N,N) ))
ax.plot_wireframe(X3,Y3, U[Local_to_global[2]].reshape( (N,N) ))
ax.plot_wireframe(X4,Y4, U[Local_to_global[3]].reshape( (N,N) ))
ax.plot_wireframe(X5,Y5, U[Local_to_global[4]].reshape( (N,N) ))
ax.plot_wireframe(X6,Y6, U[Local_to_global[5]].reshape( (N,N) ))



plt.show()







