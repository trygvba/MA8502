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
    for I in range(dofs_u):
        i = I%N
        j = I/N
        for J in range(dofs_p):
            k=J%(N-2)
            l=J/(N-2)

            if ((j==0) or (j==l) or (j==N-1)):
                # Assembling the B[I,J] part I:
                B[I,J] += weights[j]*P_evals[l,j]*(weights[0]*Y_eta[0,j]*D[i,0]*P_evals[k,0]
                        +weights[k+1]*Y_eta[k+1,j]*D[i,k+1] + weights[-1]*Y_eta[-1,j]*D[i,-1]*P_evals[k,-1])
                
                # Assembling B[I+dofs_u, J] part I:
                B[I+dofs_u, J] -= weights[j]*P_evals[l,j]*(weights[0]*X_eta[0,j]*D[i,0]*P_evals[k,0]
                                +weights[k+1]*X_eta[k+1,j]*D[i,k+1] + weights[-1]*X_eta[-1,j]*D[i,-1]*P_evals[k,-1])


            if ( (i==0) or (i==k) or (i==N-1)):
                # Assembling B[I,J] part II:
                B[I,J] -= weights[i]*P_evals[k,i]*(weights[0]*Y_xi[0,i]*D[j,0]*P_evals[l,0]
                        +weights[l+1]*Y_xi[l+1,i]*D[j,l+1] + weights[-1]*Y_xi[-1,i]*D[j,-1]*P_evals[l,-1])

                # Assembling the B[I+dofs_u, J] part II:
                B[I+dofs_u,J] += weights[i]*P_evals[k,i]*(weights[0]*X_xi[0,i]*D[j,0]*P_evals[l,0]
                                +weights[l+1]*X_xi[l+1,i]*D[j,l+1] + weights[-1]*X_xi[-1,i]*D[j,-1]*P_evals[l,-1])
   
            
    return B

