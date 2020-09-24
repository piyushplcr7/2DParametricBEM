"""
This python script is used to generate plots for the example Double Integration
of Logarithmic Kernel. It plots the function mentioned in the README file and
also plots the errors for Gaussian Quadrature when used to integrate these
functions.
"""

import numpy as np
import matplotlib.pyplot as plt

fname1 = "./../build/log_functions.txt"
fname2 = "./../build/log_integrals.txt"
data = np.loadtxt(fname1)

x1 = data[:,0]
y1 = data[:,1]
x2 = data[:,2]
y2 = data[:,3]

# Plot of the functions
plt.figure()
plt.plot(x1,y1)

plt.xlabel('y')
plt.ylabel('$\int_{-1}^{1} \log{|x-y|} dx$')

plt.title("Plot of $\int_{-1}^{1} \log{|x-y|} dx$ for y $\in$ [-1,1]")
plt.savefig("y_-1_1.eps")

plt.figure()
plt.plot(x2,y2)

plt.xlabel('y')
plt.ylabel('$\int_{-1}^{1} \log{|x-y|} dx$')

plt.title("Plot of $\int_{-1}^{1} \log{|x-y|} dx$ for y $\in$ [1,2]")
plt.savefig("y_1_2.eps")

plt.figure()
x3 = np.append(x1,x2)
y3 = np.append(y1,y2)
plt.plot(x3,y3)

plt.xlabel('y')
plt.ylabel('$\int_{-1}^{1} \log{|x-y|} dx$')

plt.title("Plot of $\int_{-1}^{1} \log{|x-y|} dx$ for y $\in$ [-1,2]")
plt.savefig("y_-1_2.eps")

data = np.loadtxt(fname2)
order = data[:,0]
error1 = data[:,1]
error2 = data[:,2]

# Plot of the Quadrature Error
plt.figure()
plt.loglog(order,error1,'b')
plt.loglog(order,error2,'r')
plt.xlabel('Quadrature Order')
plt.ylabel('Quadrature Error')
plt.legend(["y $\in$ [-1,1]","y $\in$ [1,2]"])
plt.title("Plot of Quadrature Error")
plt.savefig("quadrature_error.eps")

plt.show()
