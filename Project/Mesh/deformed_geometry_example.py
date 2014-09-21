import numpy as np
import quadrature_nodes as qn
import matplotlib.pyplot as plt
import gordon_hall as gh

#Boundary maps:
def gamma1(eta):
    theta = np.pi/4.*(eta + 1.)
    return np.cos(theta), np.sin(theta)

def gamma2(xi):
    theta = np.pi/4. * (3. - xi)
    return np.cos(theta), np.sin(theta)

def gamma3(eta):
    theta = np.pi/4. * (5. - eta)
    return np.cos(theta), np.sin(theta)

def gamma4(xi):
    theta = np.pi/4. * ( 7. + xi)
    return np.cos(theta), np.sin(theta)

##########################
Nx = 100
Ny = Nx
xis = qn.GLL_points(Nx)
etas = qn.GLL_points(Ny)

#Generate mesh using Gordon-Hall algorithm:
X, Y = gh.gordon_hall_grid(gamma1, gamma2, gamma3, gamma4, xis, etas)

#Plot all rows:
for i in range(Ny):
    plt.plot(X[i,:],Y[i,:],'r')
#Plot all columns:
for i in range(Nx):
    plt.plot(X[:,i],Y[:,i],'b')
plt.show()
