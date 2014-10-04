# Here you may see how we should build the airfoil mesh using only six(!) elements. #gethype #fuckyeah #ubermench 
import numpy as np
import quadrature_nodes as qn
import gordon_hall as gh
import structured_grids as sg
import matplotlib.pyplot as plt

###############################
#   PRELIMINARY DATA:
###############################

R = 507.79956092981
yrekt = 493.522687570593
xmax = 501.000007802345
xmin = -484.456616747519
dx = 0.001
xf = 0.2971739920760934289144667697 #x-value where f(x) has its max

x0 = R+xmin #origin of semicircle shape
xc = x0 + np.sqrt(R**2-yrekt**2) #intersection point in x; associated with yrekt
t1p = np.arcsin(yrekt/R)    #lower intersection point in t
t1m = -np.arcsin(yrekt/R)+2*np.pi #upper intersection point in t
yi = np.sqrt(R**2 - (xf -x0)**2) #absolute value of y that intersects vertical line from airfoil to circle
#angle between gamma1 and gamma3 on element 1
#theta1 = np.arccos((xc-x0)/np.sqrt((xc-x0)**2+yrekt**2))
#theta2 = np.arccos((xf-x0)/np.sqrt((xf-x0)**2+yi**2));

#Global functions:
def outer_circle(theta):
    return R*np.cos(theta)+x0, -R*np.sin(theta)

def airfoil_upper(x):
    return 0.594689181*(0.298222773*np.sqrt(x) - 0.127125232*x - 0.357907906*x**2 + 0.291984971*x**3 - 0.105174606*x**4)

def airfoil_lower(x):
    return -0.594689181*(0.298222773*np.sqrt(x) - 0.127125232*x - 0.357907906*x**2 + 0.291984971*x**3 - 0.105174606*x**4)

def theta(x,y):
    return np.arccos((x-x0)/np.sqrt((x-x0)**2+y**2))

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
def gamma_21(eta):
    return xf , 0.5*( (airfoil_upper(xf) - yi)*eta - airfoil_upper(xf)-yi )

def gamma_22(xi):
    theta1 = theta(xc,-yrekt)
    theta2 = theta(xf,-yi)
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
    theta1 = theta(xf,-yi)
    return outer_circle( (theta2-theta1)/2*xi + (theta2+theta1)/2)

def gamma_33(eta):
    return gamma_21(eta)

def gamma_34(xi):
    return (xf-(xf)*xi)/2 , airfoil_lower((xf-(xf)*xi)/2)

#Fourth element: (DONE)
def gamma_41(eta):
    return xf , 0.5*( (-airfoil_upper(xf) + yi)*eta + airfoil_upper(xf)+yi ) 
 
def gamma_42(xi):
    theta1 = np.pi
    theta2 = theta(xf,yi)+np.pi
    return outer_circle( (theta2-theta1)/2*xi + (theta2+theta1)/2)

def gamma_43(eta):
    return gamma_31(eta)

def gamma_44(xi):
    return (xf+(xf)*xi)/2 , airfoil_upper((xf+(xf)*xi)/2)

#Fifth element: (DONE)
def gamma_51(eta):
    return (xc+1)/2 + eta*(xc-1)/2 , yrekt/2*(1+eta)

def gamma_52(xi):
    theta2 = theta(xc,yrekt) +np.pi/2
    theta1 = theta(xf,yi) +np.pi/2
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

###########################################

#Order of GLL-points:
n = 500
xis = qn.GLL_points(n)

#Local to global matrix:
G = sg.local_to_global_top_down(7, 2, n)

# Dimensions of resulting matrix: (need to change)
ydim = n
xdim = (n-1)*6+1

#Total number of points:
tot_points = ydim*xdim

# Initialise coordinate matrices:
X = np.zeros( tot_points )
Y = np.zeros( tot_points )

##########################################
#LET'S GET SOME SCIENCE INDAHOUSE
##########################################

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

# Fifth element:
xtemp, ytemp = gh.gordon_hall_grid(gamma_51, gamma_52, gamma_53, gamma_54, xis, xis)
X[G[4]] = xtemp.ravel()
Y[G[4]] = ytemp.ravel()

# Sixth element:
xtemp, ytemp = gh.gordon_hall_grid(gamma_61, gamma_62, gamma_63, gamma_64, xis, xis)
X[G[5]] = xtemp.ravel()
Y[G[5]] = ytemp.ravel()


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
