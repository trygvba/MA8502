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
        P_evals[k,0] = (1.-xis[k+1]**2)*D[k+1,0]/2.
        P_evals[k,-1] = (xis[k+1]**2-1.)*D[k+1,-1]/2.

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

            if ((j==0) or (j==l+1) or (j==N-1)):
                # Assembling the B[I,J] part I:
                B[I,J] += weights[j]*P_evals[l,j]*(weights[0]*Y_eta[0,j]*D[i,0]*P_evals[k,0]
                        +weights[k+1]*Y_eta[k+1,j]*D[i,k+1] + weights[-1]*Y_eta[-1,j]*D[i,-1]*P_evals[k,-1])
                
                # Assembling B[I+dofs_u, J] part I:
                B[I+dofs_u, J] -= weights[j]*P_evals[l,j]*(weights[0]*X_eta[0,j]*D[i,0]*P_evals[k,0]
                                +weights[k+1]*X_eta[k+1,j]*D[i,k+1] + weights[-1]*X_eta[-1,j]*D[i,-1]*P_evals[k,-1])


            if ( (i==0) or (i==k+1) or (i==N-1)):
                # Assembling B[I,J] part II:
                B[I,J] -= weights[i]*P_evals[k,i]*(weights[0]*Y_xi[0,i]*D[j,0]*P_evals[l,0]
                        +weights[l+1]*Y_xi[l+1,i]*D[j,l+1] + weights[-1]*Y_xi[-1,i]*D[j,-1]*P_evals[l,-1])

                # Assembling the B[I+dofs_u, J] part II:
                B[I+dofs_u,J] += weights[i]*P_evals[k,i]*(weights[0]*X_xi[0,i]*D[j,0]*P_evals[l,0]
                                +weights[l+1]*X_xi[l+1,i]*D[j,l+1] + weights[-1]*X_xi[-1,i]*D[j,-1]*P_evals[l,-1])
   
            
    return B


def pressure_at_airfoil(K, N, loc_glob_p, P_evals, P_vec):
    """Function giving the pressure evaluated at the bottom points of
    an element.
    INPUT:
        K: Element number.
        N: Number of GLL-points.
        loc_glob_p: Local-to-global matrix for the pressure grid.
        P_evals: Basis function point evaluation matrix for the pressure basis.
        P_vec: Pressure coeffecients in the global basis.
    OUTPUT:
        P: An N-array containing the pressure extrapolated to the bottom boundary of the element.
    """
    # Initialise result vector.
    P = np.zeros(N)

    #Iterate over each basis point in the element:
    for I in range((N-2)**2):
        # x-direction:
        i = I%(N-2)
        # y-direction:
        j = I/(N-2)

        # Calculate contribution to each point:
        for k in range(N):
            #       Pressure coefficient    X-basis     Y-basis
            P[k] += P_vec[loc_glob_p[K,I]]*P_evals[i,k]*P_evals[j,0]

    return P


def calculate_lift_and_drag_contribution(X_el, Y_el, K, N, loc_glob_p, P_evals, P_vec, weights):
    """Calculates the Drag and Lift contribution from a given element.
    INPUT:
        In addition to input for pressure_at_airfoil:
        X_el, Y_el: Element grid matrices.
        weights: GLL-weights.
    OUTPUT:
        [D,L]: Array with Drag and Lift contribution.
    """
    #Start by calculating the pressure at the GLL-points:
    P = pressure_at_airfoil(K,N, loc_glob_p, P_evlas, P_vec)
    
    # Get interval end points:
    xstart = X_el[0,K]
    xend = X_el[0,K+1]

    ystart = Y_el[0,K]
    yend = Y_el[0,K+1]

    #Calculate weighted sum:
    temp = 0.5*np.dot(weights, P)
    #                     DRAG      ,   LIFT
    return temp*np.array([yend-ystar, xstart-xend])

