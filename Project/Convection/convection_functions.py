# Module containing functions for assembling the convective matrices.
import numpy as np

def calculate_convection_element(I, J, u, x_xi, x_eta, y_xi, y_eta, D, N, weights, tot_points):
    """Function for evaluating ONE element of the convection  matrix.
    INPUT:
        I,J: Indices in the matrix.
        u: Convection field, 2*tot_points elements
        x_xi, x_eta, y_xi, y_eta: The transform derivatives from the Gordon-Hall algorithm.
        D: Lagrange interpolant derivative matrix.'
        N: Number of GLL-points in each direction.
        weights: GLL-weights.
    OUTPUT:
        C[I,J]: Element of the convection matrix.
    """
    i = I%N
    j = I/N

    m = J%N
    n = J/N

    NJ_xi = (n==j)*D[m,i]
    NJ_eta = (m==i)*D[n,j]

    return weights[i]*weights[j]*(u[I]*(y_eta[j,i]*NJ_xi-y_xi[j,i]*NJ_eta) + u[I+tot_points]*(x_xi[j,i]*NJ_eta - x_eta[j,i]*NJ_xi))

def assemble_convection_matrix(u, x_xi, x_eta, y_xi, y_eta, D, N, weights):
    """Something about the function here.
    """
    tot_points = N**2
    C = np.zeros( (2*tot_points, 2*tot_points) )

    for I in range(tot_points):
        for J in range(tot_points):
            C[I,J] = calculate_convection_element(I, J, u, x_xi, x_eta, y_xi, y_eta, D, N, weights, tot_points) 

    C[tot_points:, tot_points:] = C[:tot_points,:tot_points]

    return C
