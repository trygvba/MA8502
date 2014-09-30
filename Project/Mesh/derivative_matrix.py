#Function computing the derivative matrix of a Langrange interpolant.
import numpy as np
import quadrature_nodes as qn

def diff_matrix(xis,n):
    """This function returns the derivative matrix for the Lagrange interpolation
    functions l_i(xi).
    l_i(x_k) = delta_(i,k) (Kronecker delta).
    """
    #Initialise D-matrix:
    D = np.zeros( (n,n) )
    L = np.zeros( n )
    for i in range(n):
        L[i] = qn.Legendre(xis[i],n-1)


    for i in range(n):
        for j in range(n):
            if (i==j):
                D[i,j] = 0.
            else:
                D[i,j] = L[j]/(L[i] * (xis[j] - xis[i]))

    D[0,0] = - n*(n-1)/4.
    D[-1,-1] = n*(n-1)/4.

    return D

            



