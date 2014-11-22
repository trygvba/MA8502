# Module containing functions for assembling the convective matrices.
import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse

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

    return weights[i]*weights[j]*(u[I]*(y_eta[i,j]*NJ_xi-y_xi[j,i]*NJ_eta) + u[I+tot_points]*(x_xi[j,i]*NJ_eta - x_eta[i,j]*NJ_xi))

def assemble_convection_matrix(u, x_xi, x_eta, y_xi, y_eta, D, N, weights):
    """Something about the function here.
    """
    tot_points = N**2
    C = np.zeros( (tot_points, tot_points) )

    for I in range(tot_points):
        for J in range(tot_points):
            C[I,J] = calculate_convection_element(I, J, u, x_xi, x_eta, y_xi, y_eta, D, N, weights, tot_points) 

    return C

######## NEW VERSION USED TO IMPLEMENT IN LOOPS ################ 
################################################################

def calculate_const_convection_element(I, J, x_xi, x_eta, y_xi, y_eta, D, N, weights) :
    """Function for evaluating ONE element of the convection  matrix.
    INPUT:
        I,J: Indices in the matrix.
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
    C1 = weights[i] * weights[j] * (y_eta[i,j] * NJ_xi - y_xi[j,i] * NJ_eta)
    C2 = weights[i] * weights[j] * (x_xi[j,i] * NJ_eta - x_eta[i,j] * NJ_xi)
    return C1,C2

def assemble_const_convection_matrix(x_xi, x_eta, y_xi, y_eta, D, N, weights):
    """Function that assemles the two terms in the constant convection matrix
    INPUT:
        x_xi, x_eta, y_xi, y_eta: The transform derivatives from the Gordon-Hall algorithm.
        D: Lagrange interpolant derivative matrix.'
        N: Number of GLL-points in each direction.
        weights: GLL-weights.
    OUTPUT:
        C1,C2: The two terms in the constant convection matrix.
              Each row K needs to be multiplied with the vector field U evalueted in the point K. 
              The multiplication is to be done for the x and y component of U(K) 
    """
    tot_points = N**2
    C1 = np.zeros( (tot_points, tot_points) )
    C2 = np.zeros( (tot_points, tot_points) )

    for I in range(tot_points):
        for J in range(tot_points):
            C1[I,J] , C2[I,J] = calculate_const_convection_element(I, J, x_xi, x_eta, y_xi, y_eta, D, N, weights) 

    return C1,C2
def update_convection_matrix(u1,u2,C1,C2):
  """Function for updating the convection matrix for each iteration
  INPUT:
    u1: Convection field in x direction from last iteration
    u2: Convection field in y direction from last iteration
    C1: Convection matrix without the nonlinear factor that takes care of the x-components
    C2: Convection matrix without the nonlinear factor that takes care of the y-components
  OUTPUT: 
    C: The nonlinear convection matrix
  """
  return (u1*C1.T).T , (u2*C2.T).T

def update_convection_matrix_sparse(u1,u2,C1,C2):
	"""Function for updating the sparse convection matrices.
	INPUT/OUTPUT: Same as for update_convection_matric(u1,u2,C1,C2).
	"""
	return (sparse.diags(u1,0,format='csr')*C1), (sparse.diags(u2,0, format='csr')*C2)


