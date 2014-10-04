# This module contains function to be used to assemble the stiffness matrix corresponding to the Laplacian.

import numpy as np


def G_matrix(xD, xTD, yD, yTD, alpha, beta):
    """Function returning the G-matrix for the stiffness-quadrature
    on the reference element [-1,1]^2.
    INPUT:
        xD: The local x-matrix multiplied with the lagrangian derivative matrix.
        xTD: x-matrix transposed multiplied with D.
        yD, yTD: Analogous to xD and xTD.
        alpha, beta: Integer indices determining which point should be evaluated.

    OUTPUT:
        G: a 2-by-2 matrix giving the Jacobian and gradient transform in a point.
        G = J*Delta_N(xi_alpha, xi_beta)
    """
    x_xi = xD[beta, alpha]
    x_eta = xTD[alpha, beta]
    y_xi = yD[beta, alpha]
    y_eta = yTD[alpha, beta]

    #Determining the Jacobian:
    J = np.abs(x_xi*y_eta - x_eta*y_xi)

    return 1./J * np.array( [ [x_eta**2 + y_eta**2, -(y_eta*y_xi + x_eta*x_xi)],
            [-(y_eta*y_xi + x_eta*x_xi), x_xi**2+y_xi**2] ])


def assemble_total_G_matrix(xD, xTD, yD, yTD, nx, ny):
    """Function calculating all the 2-by-2 G-matrices for all points in the element.
    INPUT:
        xD: Local x-coordinate matrix multiplied with lagrangian derivative matrix.
        xTD: The transpose of the local x-matrix and the D-matrix.
        yD, yTD: Analogously for the y-coordinates.
    OUTPUT:
        G: total G-matrix for all points in the element.
    """
    # Number of points in the element:
    num_points = nx*ny

    #Initialise the G-matrix:
    G = np.zeros( (num_points, 2, 2) )

    for I in range(num_points):
        i = I/nx
        j = I%nx
        G[I] = G_matrix(xD, xTD, yD, yTD, i, j)

    return G


def calculate_local_stiffness_element( I, J, D, G_tot, weights ):
    """ Function for calculating the I,J element of the LOCAL
    stiffness matrix. Later to be assembled with the larger global
    stiffness matrix.
    INPUT:
        I,J: Local indices.
        D: Lagrangian derivative matrix.
        G_tot: G-matrices for all points.
        weights: GLL_weights.

    OUTPUT:
        A_K[I,J]: Element of the local stiffness matrix.
    """
    # Number of GLL-points:
    N = len(weights)

    # Get x- and y- indices:
    i = I%N
    j = I/N

    m = J%N
    n = J/N

    #Initialise value:
    A = 0.
    A += D[m,i]*G_tot[j*N+m, 0, 1]*D[n,j]*weights[m]*weights[j]
    A += weights[n]*weights[i]*D[n,j]*G_tot[N*n+i, 1, 0]*D[m,i]

    if (n==j):
        for alpha in range(N):
            A += weights[alpha]*weights[j]*D[alpha,i]*G_tot[N*j+alpha, 0, 0]*D[m,alpha]

    if (m==i):
        for beta in range(N):
            A += weights[i]*weights[beta]*D[beta,j]*G_tot[N*beta+i, 1, 1]*D[n,beta]

    return A

def assemble_local_stiffness_matrix(X, Y, D, N, weights):
    """ Function for assembling the local stiffness matrix for one element
    prescribed by X and Y matrices.
    INPUT:
        X, Y: Coordinate matrices.
        D: Lagrangian derivative matrix.
        N: Number of GLL-points in each direction.
        weights: GLL-weights.
    OUTPUT:
        A_K: Local stiffness matrix for element K.
    """

    xD = np.dot(X,D)
    xTD = np.dot(X.T, D)
    yD = np.dot(y, D)
    yTD = np.dot(y.T, D)

    G_tot = assemble_total_G_matrix( xD, xTD, yD, yTD, N, N)

    # Initialise local stiffness matrix:
    num_points = N**2
    A_K = np.zeros( (num_points, num_points) )

    for I in range(num_points):
        for J in range(I+1):
            A_K[I,J] += calculate_local_stiffness_element(I, J, D, G_tot, weights)
            if (J < I):
                A_K[J,I] = A_K[I,J]

    return A_K
