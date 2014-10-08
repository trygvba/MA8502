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

###########################
# Order of GLL-points:
N = 100
xis = qn.GLL_points(N)
weights = qn.GLL_weights(N, xis)

#Local to global matrix:
G = sg.local_to_global_top_down(7, 2, N, N)

# Dimensions of resulting matrix: (need to change)
ydim = N
xdim = (N-1)*6+1

#Total number of points:
tot_points = N*N

# Initialise coordinate matrices:
X = np.zeros( tot_points )
Y = np.zeros( tot_points )


t1 = time.time()

#Generate mesh:
print "Getting Mesh..."
X1, Y1 = gh.gordon_hall_grid( gamma_11, gamma_12, gamma_13, gamma_14, xis, xis)
X[G[0]] = xtemp.ravel()
Y[G[0]] = ytemp.ravel()
X2, Y2 = gh.gordon_hall_grid( gamma_21, gamma_22, gamma_23, gamma_24, xis, xis)
X[G[1]] = xtemp.ravel()
Y[G[1]] = ytemp.ravel()
X3, Y3 = gh.gordon_hall_grid( gamma_31, gamma_32, gamma_33, gamma_34, xis, xis)
X[G[2]] = xtemp.ravel()
Y[G[2]] = ytemp.ravel()
X4, Y4 = gh.gordon_hall_grid( gamma_41, gamma_42, gamma_43, gamma_44, xis, xis)
X[G[3]] = xtemp.ravel()
Y[G[3]] = ytemp.ravel()
X5, Y5 = gh.gordon_hall_grid( gamma_51, gamma_52, gamma_53, gamma_54, xis, xis)
X[G[4]] = xtemp.ravel()
Y[G[4]] = ytemp.ravel()
X6, Y6 = gh.gordon_hall_grid( gamma_61, gamma_62, gamma_63, gamma_64, xis, xis)
X[G[5]] = xtemp.ravel()
Y[G[5]] = ytemp.ravel()

#Reshape the coordinate vector:
X = X.reshape( (ydim, xdim) )
Y = Y.reshape( (ydim, xdim) )

# Get lagrangian derivative matrix:
print "Getting derivative matrix..."
D = sg.diff_matrix(xis, N)
###############################THIS MARKS THE POINT WHERE I LITERALLY COULDN'T

# Get Jacobian and total G-matrix:
print "Getting Jacobian and G-matrices."
Jac, G_tot = lp.assemble_total_G_matrix(np.dot(X,D),
                                        np.dot(X.T,D),
                                        np.dot(Y,D),
                                        np.dot(Y.T,D),
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

###
# Assemble loading vector:
print "Assembling loading vector."
F1 = lp.assemble_loading_vector(X1, Y1, f, Jac1, weights)
F2 = lp.assemble_loading_vector(X2, Y2, f, Jac2, weights)
F3 = lp.assemble_loading_vector(X3, Y3, f, Jac3, weights)
F4 = lp.assemble_loading_vector(X4, Y4, f, Jac4, weights)
F5 = lp.assemble_loading_vector(X5, Y5, f, Jac5, weights)
F6 = lp.assemble_loading_vector(X6, Y6, f, Jac6, weights)

###Now also need a crafty Jew to assemble the global loading vector F













