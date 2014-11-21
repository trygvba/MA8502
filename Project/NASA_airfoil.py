

# Importing modules:
import sys
sys.path.insert(0, 'Mesh')
sys.path.insert(0, 'Convection')
sys.path.insert(0, 'Laplace')
sys.path.insert(0, 'Mesh/NASA')

import time

# Numpy and Scipy:
import numpy as np              #Numpy for general array creation and manipulation.
import scipy.linalg as la       #For linear algebra
import scipy.sparse as sparse   #For sparse matrices.
import scipy.sparse.linalg as sla #For linalg on sparse matrices.

# Our very own modules:
import laplace_functions as lp  #Pertaining to the Laplace operator.
import structured_grids as sg   #Function for dealing with structured grids.
import quadrature_nodes as qn   #GLL and GL nodes and weights.
import gordon_hall as gh        #Gordon-hall algorithm for deformed elements.
import convection_functions as cf#Pertaining to the convection term.
import divergence_functions as df#Pertaining to the divergence and gradient terms


# For plotting:
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
#############################################
#       LET'S GET STARTED:
#############################################

# Main parameters:
N = 4               # Number of GLL-points in each direction.
mu = 1.             # Viscosity.
alpha = np.pi/20.   # Inflow angle.
v = 1.              # Inflow velocity.
N_it = 10           # Number of iterations.
eps = 1e-8          # Error tolerance.

###################################
# Reading NASA-grid:
X_el, Y_el = sg.read_PLOT3D('Mesh/NASA_grids/grid_coarse.xyz')

# Get number of elements:
jdim, idim = X_el.shape
Nx = idim - 1
Ny = jdim - 1

# Get number of patch elements:
patches = sg.get_number_of_patch_elements(X_el, Y_el, idim)
############################################################
# Get GLL-points and -weights:
xis = qn.GLL_points(N)
weights = qn.GLL_weights(N,xis)

# Get derivative matrix:
D = sg.diff_matrix(xis,N)

# Get pressure evaluation matrix:
P_evals = df.pressure_basis_point_evaluations(xis, N, D)

# Get local-to-global mapping:
loc_glob = sg.local_to_global_top_down(idim, jdim,
                                       N, N,
                                       patch=True,
                                       num_patch=patches)


# Get local-to-global mapping for the pressure:
loc_glob_p = sg.local_to_global_pressure(idim, jdim, N, N)

# Number of elements:
num_el = loc_glob.shape[0]

# Degrees of freedom:
dofs = np.max(loc_glob)+1
dofs_p = np.max(loc_glob_p)+1

print "Number of elements: ", num_el
print "Degress of freedom velocity: ", 2*dofs
print "Dofs for pressure:", dofs_p
################################################

###########################################
#       OFF TO THE ASSEMBLY PART:
###########################################

# Initialise Important matrices:
A = sparse.lil_matrix( (dofs, dofs), dtype='float64' )            #Stiffness matrix
Cc1 = sparse.lil_matrix( (dofs, dofs), dtype='float64' )          #First constant convection matrix
Cc2 = sparse.lil_matrix( (dofs, dofs), dtype='float64' )          #Second constant convection matrix
B =  sparse.lil_matrix( (2*dofs, dofs_p), dtype='float64' )       #Divergence/gradient matrix.
#Initialse local coordinate matrices:
X = np.zeros( (num_el, N, N) )
Y = np.zeros( (num_el, N, N) )

# LET'S BEGIN ITERATING OVER EACH ELEMENT:
time_assembly = time.time()
print "Starting to assemble matrices...."
for K in range(num_el):
    #Get corner point indices for current element:
    indx = K%Nx
    indy = K/Nx

    X[K], Y[K] = gh.gordon_hall_straight_line(indy, indx, X_el, Y_el, xis, N)
    
    X_xi = np.dot(X[K],D)
    X_eta = np.dot(X[K].T,D)
    Y_xi = np.dot(Y[K], D)
    Y_eta = np.dot(Y[K].T, D)

    # Get Jacobian and total G-matrix:
    Jac, G_tot = lp.assemble_total_G_matrix(X_xi,
                                            X_eta,
                                            Y_xi,
                                            Y_eta,
                                            N,
                                            N)

    # Insert contribution to stiffness matrix:
    A[np.ix_(loc_glob[K],loc_glob[K])] += lp.assemble_local_stiffness_matrix(D, G_tot, N, weights)
    # Insert contribution to constant convection matrices:
    Cc1[np.ix_(loc_glob[K], loc_glob[K])], Cc2[np.ix_(loc_glob[K], loc_glob[K])] = \
                                            cf.assemble_const_convection_matrix(X_xi,
                                                                                X_eta,
                                                                                Y_xi,
                                                                                Y_eta,
                                                                                D,
                                                                                N,
                                                                                weights)




    # Insert contribution to divergence matrix:
    B[ np.ix_(np.hstack((loc_glob[K], loc_glob[K]+dofs)), loc_glob_p[K])] = \
            df.assemble_local_divergence_matrix(X_xi,
                                                X_eta,
                                                Y_xi,
                                                Y_eta,
                                                P_evals,
                                                D,
                                                weights,
                                                N)







print "Total assembly time: ", time.time() - time_assembly

# Convert to CSR-matrices, which are more suited for arithmetic:
A.tocsr()
Cc1.tocsr()
Cc2.tocsr()
B.tocsr()
###############################################################################################
###############################################################################################
# Create initial guess as solution of Laplace
W = sparse.bmat( [[sparse.block_diag((mu*A, mu*A)), -B],
                [ B.T, None]], format='csr')

F = np.zeros ( 2*dofs + dofs_p)
##################################################
# Function for efficiently zeroing out rows:
def csr_zero_rows(csr, rows_to_zero):
    rows, cols = csr.shape
    mask = np.ones((rows,), dtype=np.bool)
    mask[rows_to_zero] = False
    nnz_per_row = np.diff(csr.indptr)
    mask = np.repeat(mask, nnz_per_row)
    nnz_per_row[rows_to_zero] = 0
    csr.data = csr.data[mask]
    csr.indices = csr.indices[mask]
    csr.indptr[1:] = np.cumsum(nnz_per_row)
################################################


print "Imposing boundary conditions..."
time_bc = time.time()

print Nx-2*patches
# Imposing boundary conditions:
for bc in range(patches,Nx-patches):
    # Airfoil (On the bottom of the grid, if you think about it):
    indices = loc_glob[bc,:N]
    csr_zero_rows(W, indices)
    W[indices, indices] = 1.
    F[indices] = 0.
    # In y-direction:
    csr_zero_rows(W, indices+dofs)
    W[indices+dofs, indices+dofs] = 1.
    F[indices] = 0.

    # Inflow (Similarly, on the top of the grid):
    indices = loc_glob[bc+Nx*(Ny-1),-N:]
    csr_zero_rows(W, indices)
    W[indices, indices] = 1.
    F[indices] = v*np.cos(alpha)

    csr_zero_rows(W,indices+dofs)
    W[indices+dofs,indices+dofs]= 1.
    F[indices+dofs] = v*np.sin(alpha)

# Need to set on pressure point:
m = 0
csr_zero_rows(W, 2*dofs+m)
W[2*dofs+m, 2*dofs+m] = 1.
print "Time imposing boundary conditions: ", time.time()-time_bc
print "Starting to solve..."
time_solve = time.time()
UVP = sla.spsolve(W,F)
print "Time solving: ", time.time() - time_solve

##########################################################################
##########################################################################
#################       ITERATIONS:         ##############################
##########################################################################
##########################################################################

error = 1.
counter = 1

while(error>eps and counter <=N_it):
    t1 = time.time()
    print "Solving for the ", counter, "'th time."
    
    # Update convection matrices:
    C1, C2 = cf.update_convection_matrix(UVP[:dofs], UVP[dofs:2*dofs], Cc1, Cc2)
    W = sparse.bmat( [[sparse.block_diag(C1+mu*A, C2 + mu*A), -B],
                      [B.T, None]], format='csr')

    time_bc = time.time()
    # Imposing Boundary Conditions:
    for bc in range(patches, Nx-Patches):
        # Airfoil (On the bottom of the grid, if you think about it):
        indices = loc_glob[bc,:N]
        csr_zero_rows(W, indices)
        W[indices, indices] = 1.
        # In y-direction:
        csr_zero_rows(W, indices+dofs)
        W[indices+dofs, indices+dofs] = 1.

        # Inflow (Similarly, on the top of the grid):
        indices = loc_glob[bc+Nx*(Ny-1),-N:]
        csr_zero_rows(W, indices)
        W[indices, indices] = 1.

        csr_zero_rows(W,indices+dofs)
        W[indices+dofs,indices+dofs]= 1.

    csr_zero_rows(W,2*dofs+m)
    W[2*dofs+m, 2*dofs+m] = 1.
    
    print "Imposing BC:             ", time.time() - time_bc
    # Solve the system:
    time_solve = time.time()
    UVP_new = sla.spsolve(W,F)
    print "Time to solve: ", time.time() - time_solve
    # Calculate relative change from iteration:
    error = float( la.norm(UVP_new - UVP))/la.norm(UVP)
    # Update UVP-vector:
    UVP = UVP_new

    #Iterate counter:
    counter += 1
    print "Error:               ", error

   
################################################################
################################################################
##############        PLOTTING        ##########################
################################################################
################################################################

fig = plt.figure(1)
# QUIVERPLOT (No need to plot everything):
plt.subplot(121)
#Get value at each point on NASA grid:
U = np.zeros( (jdim, idim) )
V = np.zeros( (jdim, idim) )
for j in range(Ny):
    #Rightmost point:
    U[j,-1] = UVP[loc_glob[j*Nx+Nx-1,N]]
    V[j,-1] = UVP[loc_glob[j*Nx+Nx-1,N]+dofs]
    for i in range(Nx):
        #Leftmost point:
        U[-1,i] = UVP[loc_glob[Nx*(Ny-1)+i, N*(N-1)]]
        V[-1,i] = UVP[loc_glob[Nx*(Ny-1)+i, N*(N-1)]+dofs]

        #General point:
        U[j,i] = UVP[loc_glob[Nx*j+i,0]]
        V[j,i] = UVP[loc_glob[Nx*j+i,0]+dofs]

#Right- and topmost corner:
U[-1,-1] = UVP[loc_glob[-1,-1]]
V[-1,-1] = UVP[loc_glob[-1,-1]+dofs]

pl.quiver(X_el, Y_el, U, V)

pl.xlabel('$x$')
pl.ylabel('$y$')
pl.axis('image')

# PRESSURE PLOT:
plt.subplot(122)
ax = fig.add_subplot(122, projection='3d')
for K in range(num_el):
    ax.plot_wireframe( X[K, 1:-1, 1:-1], Y[K, 1:-1, 1:-1], UVP[loc_glob_p[K]+2*dofs].reshape((N-2,N-2)))

pl.show()



