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
