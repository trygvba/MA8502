# Numpy and Scipy:
import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse

##############################################
# GENERATE MESH
##############################################

##mesh inputs
thetadef = 25 # angle between elements 3,4 etc
thetadef = thetadef*np.pi/180
num_el = 6 # Number of elements

#constants
R = 507.79956092981
yrekt = 493.522687570593
xmax = 501.000007802345
xmin = -484.456616747519
dx = 0.001
xf = 0.2971739920760934289144667697 #x-value where f(x) has its max
##angle from xf to define elements 3 and 4 (in radians)



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

def gammas(index,number): #index corresponds to element, number to the curve number [1,4]
	if index==0:
		if number==1:
			return gamma_11
		elif number==2:
			return gamma_12
		elif number==3:
			return gamma_13
		elif number==4:
			return gamma_14		
	elif index==1:
		if number==1:
			return gamma_21		
		elif number==2:
			return gamma_22		
		elif number==3:
			return gamma_23		
		elif number==4:
			return gamma_24		
	elif index==2:
		if number==1:
			return gamma_31		
		elif number==2:
			return gamma_32		
		elif number==3:
			return gamma_33		
		elif number==4:
			return gamma_34
	elif index==3:
		if number==1:
			return gamma_41		
		elif number==2:
			return gamma_42		
		elif number==3:
			return gamma_43		
		elif number==4:
			return gamma_44
	elif index==4:
		if number==1:
			return gamma_51		
		elif number==2:
			return gamma_52		
		elif number==3:
			return gamma_53		
		elif number==4:
			return gamma_54
	elif index==5:
		if number==1:
			return gamma_61		
		elif number==2:
			return gamma_62		
		elif number==3:
			return gamma_63
		elif number==4:
			return gamma_64
	else:
		return -1

