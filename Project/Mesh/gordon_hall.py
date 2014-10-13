import numpy as np

def gordon_hall_grid(gamma1, gamma2, gamma3, gamma4, xis, etas):
    """This quite extensive function generates the mesh for a deformed geometry,
    given the boundary mappings (gammas).
    INPUT:
        gammas: Functions or function handles that describe the boundary.
        xis: GLL-points on the reference square in the x-direction.
        etas: GLL-points in the y-direction.
    OUTPUT: Mesh matrices X and Y.

    NOTE: the numbering of boundary sides are important. Be careful to use the following
    convention:
        gamma1: x(1,etas),
        gamma2: x(xis,1),
        gamma3: x(-1,etas),
        gamma4: x(xis, -1).
    """
    Nx = len(xis)
    Ny = len(etas)

    #Interpolants:
    phi0 = lambda r: (1-r)/2.
    phi1 = lambda r: (1+r)/2.


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
            xC[m,n] = (phi0(etas[m])*(phi0(xis[n])*xB[0,0] + phi1(xis[n])*xB[0,-1]) +
                    phi1(etas[m])*(phi0(xis[n])*xB[-1,0] + phi1(xis[n])*xB[-1,-1]) )
            yC[m,n] = (phi0(etas[m])*(phi0(xis[n])*yB[0,0] + phi1(xis[n])*yB[0,-1]) +
                    phi1(etas[m])*(phi0(xis[n])*yB[-1,0] + phi1(xis[n])*yB[-1,-1]) )

    #Total assembly and plot:
    X = xA + xB - xC
    Y = yA + yB - yC
    return X, Y


def gordon_hall_straight_line(i,j, X_glob, Y_glob, xis, N):
    """Implementation for the Gordon-Hall algorithm, where the
    boundary mappings are assumed to be straight lines, i.e. a simpler case.
    INPUT:
        i,j: Indices for one corner point of the four-sided shape.
            Note that i corresponds to the y-dimension.
        X_glob, Y_glob: Global coordinate matrices.
        xis: GLL-nodes.
        N: Number of GLL-points.
    OUTPUT:
        X_loc, Y_loc: Local coordinate matrices.
    """
    # Initialise the matrices:
    X_loc = np.zeros( (N,N) )
    Y_loc = np.zeros( (N,N) )

    #Put in the corner points:
    X_loc[0,0] = X_glob[i,j]
    Y_loc[0,0] = Y_glob[i,j]

    X_loc[0,-1] = X_glob[i,j+1]
    Y_loc[0,-1] = Y_glob[i,j+1]

    X_loc[-1,0] = X_glob[i+1,j]
    Y_loc[-1,0] = Y_glob[i+1,j]

    X_loc[-1,-1] = X_glob[i+1,j+1]
    Y_loc[-1,-1] = Y_glob[i+1,j+1]

    #Interpolate the corner points:
    X_loc[0] = X_loc[0,0]*(1-xis)/2. + X_loc[0,-1]*(1-xis)/2.
    Y_loc[0] = Y_loc[0,0]*(1-xis)/2. + Y_loc[0,-1]*(1-xis)/2.

    X_loc[-1] = X_loc[-1,0]*(1-xis)/2. + X_loc[-1,-1]*(1+xis)/2.
    Y_loc[-1] = Y_loc[-1,0]*(1-xis)/2. + Y_loc[-1,-1]*(1+xis)/2.

    for k in range(N):
        X_loc[:,k] = X_loc[0,k]*(1-xis)/2. + X_loc[-1,k]*(1+xis)/2.
        Y_loc[:,k] = Y_loc[0,k]*(1-xis)/2. + Y_loc[-1,k]*(1+xis)/2.

    return X_loc, Y_loc
        
