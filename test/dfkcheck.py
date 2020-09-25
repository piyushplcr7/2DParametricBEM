import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

fname1 = "neumann.txt"
#fname1 = "convergence.txt"
data = np.abs(np.loadtxt(fname1))
phiex = np.linspace(0,2*np.pi,100)
srcpt = np.array([2,0])
tnex = np.array([])

for phi in phiex:
    x = np.array([np.cos(phi), np.sin(phi)])
    dotprod = np.dot(srcpt-x,x)/(LA.norm(srcpt-x)**2)
    tnex = np.append(tnex,dotprod /2./np.pi)

plt.figure()
#plt.title('single layer')
#plt.loglog(data[:,0],data[:,1])
#plt.semilogy(data[:,0],data[:,1])
#plt.xlabel('Order')
#plt.ylabel('Error')
plt.plot(phiex,tnex)
id = 5
print(data[id-3,:])
plt.plot(np.linspace(0,2*np.pi,id), data[id-3,1:1:id])
#plt.savefig('convergence.eps')
plt.show()
