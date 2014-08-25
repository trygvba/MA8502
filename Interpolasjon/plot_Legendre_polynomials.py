#This is intended as a short script for plotting the Legendre polynomials
#up to some order n.

import numpy as np
import matplotlib.pyplot as plt

#max order of polynomials:
N = 5

#Starting by assembling the coeffecients matrix:
coeffs = np.zeros ( (N+1,N+1) )
coeffs[0,0] = 1
coeffs[1,1] = 1

if N>1:
	for n in range(1,N):
		temp = 1/float(n+1)
		for j in range(N+1):
			#Using recursion relation to establish coefficients:
			coeffs[n+1,j] = temp *((2*n+1)*coeffs[n,j-1] - n*coeffs[n-1,j])


#Establishing "point"-matrix:
#Number of points:
num_points = 100
x = np.linspace(-1,1, num_points)

P = np.zeros ( (N+1,num_points) )
P[0,:] = 1

for i in range(1,N+1):
	P[i,:] = x ** i

#Now we can start thinking about plotting:
#Polynomials evaluated in points given by x:
Y = np.dot ( coeffs, P )

for i in range(N+1):
	plt.plot(x,Y[i,:], label=str(i))

plt.show()
