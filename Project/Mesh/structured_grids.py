# A collection of functions for handling the structured grid.

import numpy as np
import quadrature_nodes as qn
import gordon_hall as gh

def diff_matrix(xis,n):
    """This function returns the derivative matrix for the Lagrange interpolation
    INPUT:
        xis: Array with GLL quadrature points on the reference interval [-1,1].
        n: Number of quadrature points. Must be len(xis).
    
    OUTPUT:
        D: Derivative matrix where D[i,j] = l'_j(xis[j]).
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

 
def local_to_global_matrix( idim, jdim, n, patch=False, patch_elements=0):
    """Function returning a local-to-global mapping for a regular
    structured grid (no patching as done in the NASA grid).

    INPUT:
        idim: Number of nodes in the x-direction.
        jdim: Number of nodes in the y-direction.
        n: Number of GLL-points for each dimension in each element.
        patch: Boolean value saying if any of the bottom elements are to be
            patched together at the lower boundary symmetrically.
        patch_elements: If patch=True, how many elements are to be patched?
            Requires patch_elements <= (idim-1)/2

    OUTPUT:
        G: a (idim-1)*(jdim-1) times n**2 integer matrix.
         Each row gives the global indices for every degree of freedom
         for that element. I.e. one row corresponds to one element.
    """
    # Check that number of points in each direction is valid:
    if ((idim < 2) or (jdim < 2)):
        print "Not valid number of points in either x- or y-direction."
        return -1

    # Check if patch parameters are valid:
    if (patch):
        if (patch_elements > (idim-1)/2):
            print "The patch_element value passed is not valid."
            return -1

    # Number of elements in the x-direction:
    Nx = idim-1
    # Number of elements in the y-direction:
    Ny = jdim-1

    # Number of elements:
    num_el = Nx * Ny

    # Number of degrees of freedom for each element:
    dof_loc = n ** 2

    # Initialise local-to-global matrix:
    # (Intialised to -1 since to index will have negative value.)
    G = -np.ones( (num_el, dof_loc), dtype='int' )

    # Start off by setting the first row:
    G[0,:] = range(dof_loc)
    
    # The right boundary of the zero'th element is the
    # left boundary  of the next element:
    if (Nx > 1):
        for i in range(n):
            G[1,n*i] = G[0,n*i+n-1]

    # The upper boundary of the zero'th element is the
    # lower boundary of idim'th element:
    if (Ny > 1):
        for i in range(n):
            G[Nx, i] = G[0, n*(n-1)+i]

    # The top-right corner of the zero'th element is the
    # first dof of the (idim+1)'th element:
    if ( min(Nx, Ny) > 1):
        G[Nx+1, 0] = G[0,dof_loc-1] 

    # If patch=True and patch_elements>=1, patch the first and last element
    # on the bottom row:
    if ( (patch) and (patch_elements > 0) ):
        G[Nx-1,:n] = G[0,(n-1)::-1]
        if (patch_elements == 1):
            G[Nx-2, n-1] = G[Nx-1, 0]

    #Setting the index counter:
    current_dof = dof_loc

    #Iterating over the remaining elements:
    for i in range(1,num_el):
        #Set all new dofs in that element:
        for j in range(dof_loc):
            if (G[i,j] < 0):
                G[i,j] = current_dof
                current_dof += 1

        #Patch, if necessary:
        if ( (patch) and (i<patch_elements) and (i<Nx) ):
            G[Nx-1-i,:n] = G[i,(n-1)::-1]

        # The (Nx-2-patch_elements)'th element is also affected:
        if ( (patch) and (i==patch_elements-1) and (G[Nx-patch_elements-1,n-1] <0) ):
            G[Nx-patch_elements-1, n-1] = G[Nx-patch_elements,0] 

        # If we're not at a rightmost element, we set the left boundary
        # of the next element:
        if not((i+1)%Nx == 0):
            for k in range(n):
                if (G[i+1,k*n] < 0):
                    G[i+1, k*n] = G[i, k*n + n - 1]


        # If we're not on a topmost element, we set the lower boundary
        # of the element on top of the current element, to this ones'
        # upper boundary (...):
        if ( i/Nx < Ny-1 ):
            for k in range(n):
                if (G[i+Nx,k] < 0 ):
                    G[i+Nx, k] = G[i, n*(n-1) + k]
        
        # If we're on neither a top- or rightmost element:
        if ( (i/Nx < Ny-1) and ((i+1)%Nx > 0) ):
            if (G[i+Nx+1, 0] < 0):
                G[i+Nx+1, 0] = G[i, dof_loc -1]


    return G


def local_to_global_top_down(idim, jdim, n):
    """ Similar to local_to_global, only now the data structure
    makes it easier to plot the resulting grid.

    INPUT:
        idim: Number of points in the x-direction.
        jdim: Number of points in the y-direction.
        n: Number of GLL-points for each dimension and element.
    OUTPUT:
        G: Local-to-global matrix.
    """
    
    #Number of elements in x-direction:
    Nx = idim-1
    #Nuber of elements in y-direction:
    Ny = jdim-1

    #Number of elements:
    num_el = Nx * Ny

    #Calculating total number of points:
    np_x = Nx*(n-1) + 1
    np_y = Ny*(n-1) + 1

    tot_points = np_x * np_y
    
    grid_indices = np.arange(tot_points).reshape ( (np_y, np_x) )

    #Now we need to unravel this matrix and put it correctly into out G-matrix:
    G = np.zeros( (num_el, n**2), dtype='uint32' )
    for k in range(num_el):
        #Start indices:
        i = (n-1)*(k/Nx)
        j = (n-1)*(k%Nx)

        #Unravel indices:
        G[k,:] = grid_indices[i:(i+n), j:(j+n)].ravel()


    return G

