
import numpy as np
import matplotlib.pyplot as plt
##########################

alpha = np.pi*np.array( [-1./12, -1./15., -1./20., 0., 1./20., 1./15., 1./12.] )

D = np.array ( [0.0208233, 0.025676, 0.03044, 0.0383, 0.03044, 0.025676, 0.0208233])
L = np.array ( [-0.347916, -0.29628, -0.2339, -5.6e-8, 0.2339, 0.29628, 0.347916])

Lrot = np.zeros( len(L) )
Drot = np.zeros( len(D) )

for i in range(len(L)):
    Drot[i] = np.cos(alpha[i])*D[i] + np.sin(alpha[i])*L[i]
    Lrot[i] = -np.sin(alpha[i])*D[i] + np.cos(alpha[i])*L[i]



#Make plots:
fig = plt.figure(1)

plt.subplot(121)
plt.plot(alpha, Lrot, 'r')
plt.xlabel(r'$\alpha$')
plt.ylabel('Lift')
plt.title('Lift')

plt.subplot(122)
plt.plot(alpha, Drot, 'r')
plt.xlabel(r'$\alpha$')
plt.ylabel('Drag')
plt.title('Drag')

plt.show()

