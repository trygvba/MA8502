def calculate_convection_element(I, J, x_xi, x_eta, y_xi, y_eta, D, N, weights, tot_points):
    """Function for evaluating ONE element of the convection  matrix.
    INPUT:
        I,J: Indices in the matrix.
        x_xi, x_eta, y_xi, y_eta: The transform derivatives from the Gordon-Hall algorithm.
        D: Lagrange interpolant derivative matrix.
        N: Number of GLL-points in each direction.
        weights: GLL-weights.
    OUTPUT:
        stuff
    TODO:
    	the functions N_p in an orderly manner
    """
    i = I%N
    j = I/N

    k = J%N
    l = J/N

    #NJ_xi = (n==j)*D[m,i]
    #NJ_eta = (m==i)*D[n,j]
    
    def N_p(arg):
    	return 0
    
    #need to define the functions N_p

    if I<N**2:
	    tmp = 0
	    for alpha in range (0,N):
	    	tmp += weights[j]*weights[alpha]*N_p(xi_alpha)*N_p(eta_j)*y_eta[alpha,j]*D[i,alpha]
	    for beta in range (0,N):
	    	tmp += -weights[beta]*weights[i]*N_p(xi_i)*N_p(eta_beta)*y_eta[i,beta]*D[j,beta]
	    return tmp
	    #return weights[i]*weights[j]*(u[I]*(y_eta[i,j]*NJ_xi-y_xi[j,i]*NJ_eta) + u[I+tot_points]*(x_xi[j,i]*NJ_eta - x_eta[i,j]*NJ_xi))
    else: #I>=N**2
	    tmp = 0
	    for alpha in range (0,N):
	    	tmp += -weights[j]*weights[alpha]*N_p(xi_alpha)*N_p(eta_j)*x_eta[alpha,j]*D[i,alpha]
	    for beta in range (0,N):
	    	tmp += weights[beta]*weights[i]*N_p(xi_i)*N_p(eta_beta)*x_eta[i,beta]*D[j,beta]
	    return tmp
#return weights[i]*weights[j]*(u[I]*(y_eta[i,j]*NJ_xi-y_xi[j,i]*NJ_eta) + u[I+tot_points]*(x_xi[j,i]*NJ_eta - x_eta[i,j]*NJ_xi))
