import numpy as np
import quadrature_nodes as qn
import matplotlib.pyplot as plt

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
####################################
#Interpolants:
def phi0(r):
    return (1-r)/2.

def phi1(r):
    return (1+r)/2.
####################################
Nx = 100
Ny = Nx
xis = qn.GLL_points(Nx)
etas = qn.GLL_points(Ny)


#constructing A-part:
xA = np.zeros( (Ny,Nx) )
yA = np.zeros( (Ny,Nx) )

xA[:,Nx-1], yA[:,Nx-1] = gamma1(etas)
xA[:,0], yA[:,0] = gamma3(etas)

for i in range(Ny):
    xA[i,:] = (1-xis)/2.*xA[i,0] + (1+xis)/2.*xA[i,-1]
    yA[i,:] = (1-xis)/2.*yA[i,0] + (1+xis)/2.*yA[i,-1]

#constructing the B-part:
xB = np.zeros( (Ny,Nx) )
yB = np.zeros( (Ny,Nx) )

xB[0,:], yB[0,:] = gamma4(xis)
xB[-1,:], yB[-1,:] = gamma2(xis)

for i in range(Nx):
    xB[:,i] = (1-etas)/2.*xB[0,i] + (1+etas)/2.*xB[-1,i]
    yB[:,i] = (1-etas)/2.*yB[0,i] + (1+etas)/2.*yB[-1,i]

#Constructing the C-part:
xC = np.zeros( (Ny,Nx) )
yC = np.zeros( (Ny,Nx) )

for m in range(Ny):
    for n in range(Nx):
        xC[m,n] = (phi0(etas[m])*(phi0(xis[n])*xB[0,0] + phi1(xis[n])*xB[0,-1]) + phi1(etas[m])*(phi0(xis[n])*xB[-1,0] + phi1(xis[n])*xB[-1,-1]) )
        yC[m,n] = phi0(etas[m])*(phi0(xis[n])*yB[0,0] + phi1(xis[n])*yB[0,-1]) + phi1(etas[m])*(phi0(xis[n])*yB[-1,0] + phi1(xis[n])*yB[-1,-1])

#Total assembly and plot:
X = xA + xB - xC
Y = yA + yB - yC

#Plot all rows:
for i in range(Ny):
    plt.plot(X[i,:],Y[i,:],'r')
#Plot all columns:
for i in range(Nx):
    plt.plot(X[:,i],Y[:,i],'b')
plt.show()
