#This example is to construct a simple grid containing four elements, and arbitrary
# order of GLL-points in each element.
import numpy as np
import quadrature_nodes as qn
import structured_grids as sg
import gordon_hall as gh

import matplotlib.pyplot as plt

#We will be doing it on an annulus with four elements.


###################################
#   BOUNDARY MAPS:
###################################
# First element:
def gamma_11(eta):
    return 0., -0.5*(3. + eta)

def gamma_12(xi):
    theta = -np.pi/4. * (xi+1.)
    return 2.*np.cos(theta), 2.*np.sin(theta)

def gamma_13(eta):
    return 0.5*(3.+eta), 0.

def gamma_14(xi):
    theta = -np.pi/4. * (xi+1.)
    return np.cos(theta), np.sin(theta)

# Second element:
def gamma_21(eta):
    return -0.5*(eta+3.), 0.

def gamma_22(xi):
    theta = np.pi/4. * (5.-xi)
    return 2.*np.cos(theta), 2.*np.sin(theta)

def gamma_23(eta):
    return gamma_11(eta)

def gamma_24(eta):
    theta = np.pi/4. * (5.-eta)
    return np.cos(theta), np.sin(theta)

# Third element:
def gamma_31(eta):
    return 0., 0.5*(eta+3.)

def gamma_32(xi):
    theta = np.pi/4. * (3.-xi)
    return 2.*np.cos(theta), 2.*np.sin(theta)

def gamma_33(eta):
    return gamma_21(eta)

def gamma_34(xi):
    theta = np.pi/4. * (3.-xi)
    return np.cos(theta), np.sin(theta)

# Fourth, and last, element:
def gamma_41(eta):
    return gamma_13(eta)

def gamma_42(xi):
    theta = np.pi/4. * (1.-xi)
    return 2.*np.cos(theta), 2.*np.sin(theta)

def gamma_43(eta):
    return gamma_31(eta)

def gamma_44(xi):
    theta = np.pi/4. * (1.-xi)
    return np.cos(theta), np.sin(theta)

####################################################

#GLL-order:
n = 20
xis = qn.GLL_points(n)

# Dimensions of resulting matrix:
ydim = n
xdim = (n-1)*4+1

#Total number of points:
tot_points = ydim*xdim

#Set local-to-global matrix:
G = sg.local_to_global_top_down( 5, 2, n, n)

# Initialise coordinate matrices:
X = np.zeros( tot_points )
Y = np.zeros( tot_points )

#Let's try to insert the global coordinates coorectly:

# First element:
xtemp, ytemp = gh.gordon_hall_grid(gamma_11, gamma_12, gamma_13, gamma_14, xis, xis)


X[G[0]] = xtemp.ravel()
Y[G[0]] = ytemp.ravel()

# Second element:
xtemp, ytemp = gh.gordon_hall_grid(gamma_21, gamma_22, gamma_23, gamma_24, xis, xis)
X[G[1]] = xtemp.ravel()
Y[G[1]] = ytemp.ravel()

# Third element:
xtemp, ytemp = gh.gordon_hall_grid(gamma_31, gamma_32, gamma_33, gamma_34, xis, xis)
X[G[2]] = xtemp.ravel()
Y[G[2]] = ytemp.ravel()

# Fourth element:
xtemp, ytemp = gh.gordon_hall_grid(gamma_41, gamma_42, gamma_43, gamma_44, xis, xis)
X[G[3]] = xtemp.ravel()
Y[G[3]] = ytemp.ravel()


#Reshape the coordinate vector:
X = X.reshape( (ydim, xdim) )
Y = Y.reshape( (ydim, xdim) )

# Plot the matrix and see what sticks:

# Plot rows:
for i in range(ydim):
    plt.plot( X[i,:], Y[i,:], 'r')


# Plot columns:
for i in range(xdim):
    plt.plot( X[:,i], Y[:,i], 'b')

plt.show()
