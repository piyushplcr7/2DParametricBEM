import numpy as np
import matplotlib.pyplot as plt

fname1 = "convergence.txt"
data = np.loadtxt(fname1)

plt.figure()
plt.title('single layer')
plt.loglog(data[:,0],data[:,1])
plt.xlabel('order')
plt.ylabel('error')

plt.figure()
plt.title('Double layer')
plt.loglog(data[:,0],data[:,2])
plt.xlabel('order')
plt.ylabel('error')

plt.show()
