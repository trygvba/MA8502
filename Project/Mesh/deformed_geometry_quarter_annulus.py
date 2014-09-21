import numpy as np
import quadrature_nodes as qn
import gordon_hall as gh
import matplotlib.pyplot as plt

#Boundary mappings:
def gamma1(eta):
    theta = np.pi/4.*(eta+1)
    return 2.*np.cos(theta), 2.*np.sin(theta)

def gamma2(xi):
    return 0., 0.5*(3.+xi)

def gamma3(eta):
    theta = np.pi/4.*(eta+1)
    return np.cos(theta), np.sin(theta)

def gamma4(xi):
    return 0.5*(3.+xi), 0.


############################################

#Number of GLL points:
N = 50
xis = qn.GLL_points(N)
etas = xis

#Generate mesh:
X, Y = gh.gordon_hall_grid(gamma1, gamma2, gamma3, gamma4, xis, etas)

#Plot all rows:
for i in range(N):
    plt.plot(X[i,:],Y[i,:],'b')
    plt.plot(X[:,i],Y[:,i],'b')

plt.show()

