import numpy as np
import quadrature_nodes as qn
import matplotlib.pyplot as plt

n = 100
x = qn.GLL_points(n)

X, Y = np.meshgrid(x,x)

for i in range(n):
    plt.plot(X[:,i], Y[:,i], 'r')
    plt.plot(X[i,:], Y[i,:], 'b')
plt.show()
