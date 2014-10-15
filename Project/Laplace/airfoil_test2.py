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
Nx = 100
Ny = 100
xis = qn.GLL_points(Nx)
etas = qn.GLL_points(Ny)
weights_x = qn.GLL_weights(Nx, xis)
weights_y = qn.GLL_weights(Ny, etas)

t1 = time.time()

#Generate mesh:
print "Getting Mesh..."
X1, Y1 = gh.gordon_hall_grid( gamma_11, gamma_12, gamma_13, gamma_14, xis, etas)
X2, Y2 = gh.gordon_hall_grid( gamma_21, gamma_22, gamma_23, gamma_24, xis, etas)
X3, Y3 = gh.gordon_hall_grid( gamma_31, gamma_32, gamma_33, gamma_34, xis, etas)
X4, Y4 = gh.gordon_hall_grid( gamma_41, gamma_42, gamma_43, gamma_44, xis, etas)
X5, Y5 = gh.gordon_hall_grid( gamma_51, gamma_52, gamma_53, gamma_54, xis, etas)
X6, Y6 = gh.gordon_hall_grid( gamma_61, gamma_62, gamma_63, gamma_64, xis, etas)

# Get lagrangian derivative matrix:
print "Getting derivative matrix..."
Dx = sg.diff_matrix(xis, Nx)
Dy = sg.diff_matrix(etas, Ny)

# Get Jacobian and total G-matrix:
print "Getting Jacobian and G-matrices."
Jac1, G_tot1 = lp.assemble_total_G_matrix(np.dot(X1,Dx),
                                        np.dot(X1.T,Dx),
                                        np.dot(Y1,Dy),
                                        np.dot(Y1.T,Dy),
                                        Nx,
                                        Ny)
Jac2, G_tot2 = lp.assemble_total_G_matrix(np.dot(X2,Dx),
                                        np.dot(X2.T,Dx),
                                        np.dot(Y2,Dy),
                                        np.dot(Y2.T,Dy),
                                        Nx,
                                        Ny)
Jac3, G_tot3 = lp.assemble_total_G_matrix(np.dot(X3,Dx),
                                        np.dot(X3.T,Dx),
                                        np.dot(Y3,Dy),
                                        np.dot(Y3.T,Dy),
                                        Nx,
                                        Ny)
Jac4, G_tot4 = lp.assemble_total_G_matrix(np.dot(X4,Dx),
                                        np.dot(X4.T,Dx),
                                        np.dot(Y4,Dy),
                                        np.dot(Y4.T,Dy),
                                        Nx,
                                        Ny)
Jac5, G_tot5 = lp.assemble_total_G_matrix(np.dot(X5,Dx),
                                        np.dot(X5.T,Dx),
                                        np.dot(Y5,Dy),
                                        np.dot(Y5.T,Dy),
                                        Nx,
                                        Ny)
Jac6, G_tot6 = lp.assemble_total_G_matrix(np.dot(X6,Dx),
                                        np.dot(X6.T,Dx),
                                        np.dot(Y6,Dy),
                                        np.dot(Y6.T,Dy),
                                        Nx,
                                        Ny)


#these functions probably need to be changed to allow for different amount of nodes in x,y                                        
print "Assembling stiffness matrix."
A1 = lp.assemble_local_stiffness_matrix(D, G_tot1, N, weights)
A2 = lp.assemble_local_stiffness_matrix(D, G_tot2, N, weights)
A3 = lp.assemble_local_stiffness_matrix(D, G_tot3, N, weights)
A4 = lp.assemble_local_stiffness_matrix(D, G_tot4, N, weights)
A5 = lp.assemble_local_stiffness_matrix(D, G_tot5, N, weights)
A6 = lp.assemble_local_stiffness_matrix(D, G_tot6, N, weights)


#def assemble_local_stiffness_matrix2(Dx,Dy, G_tot, Nx,Ny, weightsx,weightsy):
    ##Dafuq? Functions like this you should put in another folder.
###Now need a crafty Jew to assemble the global stiffness matrix A

Address = local_to_global_top_down(7, 2, Nx, Ny, patch=True, num_pacht=1)


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













