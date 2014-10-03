# Here you may see how we should build the airfoil mesh using only six(!) elements.
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
t1p = np.asin(yrekt/R)    #lower intersection point in t
t1m = -np.asin(yrekt/R)+2*np.pi #upper intersection point in t
yi = np.sqrt(R**2 - (xf -x0)**2) #absolute value of y that intersects vertical line from airfoil to circle
#angle between gamma1 and gamma3 on element 1
theta1 = np.acos((xc-x0)/np.sqrt((xc-x0)**2+yrekt**2))
theta2 = np.acos((xf-x0)/np.sqrt((xf-x0)**2+yi**2));

#Global functions:
def outer_circle(theta):
    return R*np.cos(theta)+x0, R*np.sin(theta)


###############################
#   BOUNDARY MAPS:
###############################

#First element:
def gamma_11(eta):
    return

def gamma_12(xi):
    return

def gamma_13(eta):
    return

def gamma_14(xi):
    return

#Second element:
def gamma_21(eta):
    return

def gamma_22(xi):
    return

def gamma_23(eta):
    return gamma_11(eta)

def gamma_24(xi):
    return

#Third element:
def gamma_31(eta):
    return

def gamma_32(xi):
    return

def gamma_33(eta):
    return gamma_21(eta)

def gamma_34(xi):
    return


#Fourth element:
def gamma_41(eta):
    return

def gamma_42(xi):
    return

def gamma_43(eta):
    return gamma_31(eta)

def gamma_44(xi):
    return

#Fifth element:
def gamma_51(eta):
    return

def gamma_52(xi):
    return

def gamma_53(eta):
    return gamma_41(eta)

def gamma_54(xi):
    return

#Sixth element:
def gamma_61(eta):
    return

def gamma_62(xi):
    return

def gamma_63(eta):
    return gamma_51(eta)

def gamma_64(xi):
    return

###########################################

#Order of GLL-points:
n = 5
xis = qn.GLL_points(n)

#Local to global matrix:
G = sg.local_to_global_top_down(7, 2, n)

