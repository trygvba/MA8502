# Module containing functions for handling the divergence terms in the system.
import numpy as np

def pressure_basis_point_evaluations(xis, N, D):
    """Function for evaluating the pressure basis functions
    over all points in the reference interval [-1,1].
    The only real challenge lies in evaluating at the boundary
    points.
    INPUT:
        xis: GLL-points.
        N: Number of GLL-points,
        D: Langrange interpolant derivative matrix for the GLL basis points.
    OUTPUT:
        P_evals: Pressure basis evaluated at each point.
            P_k(xis[i]) = P_evals[k,i]
    """
    # Number of basis functions:
    Np = N-2

    P_evals = np.zeros( (Np, N) )
    for k in range(Np):
        # Set the kronecker delta:
        P_evals[k,k+1] = 1.
        # Set the boundary points:
        P_evals[k,0] = (1.-xis[k]**2)*D[k,0]/2.
        P_evals[k,-1] = (xis[k]**2-1)*D[k,-1]/2.

    return P_evals


def assemble_local_divergence_matrix(X_xi, X_eta, Y_xi, Y_eta, P_evals, D, weights, N):
    """Function assembling the local divergence/gradient matrix.
    INPUT:
        X_xi, X_eta, Y_xi, Y_eta: Mapping from reference element derivatives.
        P_evals: The pressure basis functions evaluated at the GLL-nodes.
        D: Lagrange interpolant derivative matrix.
        weights: GLL-weights.
        N: Number of GLL-points.
    OUTPUT:
        B: The divergence/gradient matrix (dimension  2N**2 x M)
    """
    dofs_p = (N-2)**2
    dofs_u = N**2

    #Initialise the B-matrix:
    B = np.zeros( (2*dofs_u, dofs_p) )

    #Starting the assembly phase:
    for J in range(dofs_p):
        k = J%(N-2)
        l = J/(N-2)
        for I in range(dofs_u):
            i = I%N
            j = I/N
            # Assembling B[I,J]:
            B[I,J] = np.sum( weights[j]*weights*P_evals[k]*P_evals[l,j]*Y_eta[:,j]*D[i,:] )
            B[I,J] += -np.sum( weights[i]*weights*P_evals[k,i]*P_evals[l]*Y_xi[:,i]*D[j,:] )

            # Assembling B[I+N, J]
            B[I+dofs_u, J] = -np.sum( weights[j]*weights*P_evals[k]*P_evals[l,j]*X_eta[:,j]*D[i,:] )
            B[I+dofs_u, J] += np.sum( weights[i]*weights*P_evals[k,i]*P_evals[l]*X_xi[:,i]*D[j,:] )

    return B
