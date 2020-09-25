"""
This python script generates a plot comparing the exact neumann traces and
the traces calculated in annular_dvp.cpp
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

# Loading the data
fname2 = "neumann2dpbem.txt"
rawdata2 = np.loadtxt(fname2)

fname3 = "tnpbemex.txt"
rawdata3 = np.loadtxt(fname3)

def wtfmani(phi):
    normal = np.array([1*np.cos(phi),2*np.sin(phi)])
    return normal/LA.norm(normal)

# Function to plot the neumann trace for the inner boundary
def plotinner(i):
    data = rawdata2
    # number of panels at position i in the file
    n_items = 1+3*i
    # calculated traces
    inner = data[i,0:n_items]
    # corresponding theta values
    theta = np.linspace(0,2*np.pi,n_items+1)[0:n_items]
    theta = theta + (np.pi/n_items)
    # exact traces
    inner_ex = rawdata3[i,0:n_items]
    # plotting the values
    plt.plot(theta,inner)
    plt.plot(theta,inner_ex)
    plt.legend(['evaluated','exact'])

# Function to plot the neumann trace for the outer boundary
def plotouter(i):
    data = rawdata2
    # number of panels at position i in the file
    n_items = 1+3*i
    # calculated traces
    outer = data[i,n_items:2*n_items]
    # corresponding theta values
    theta = np.linspace(0,2*np.pi,n_items+1)[0:n_items]
    theta = theta + (np.pi/n_items)
    # exact traces
    outer_ex = rawdata3[i,n_items:2*n_items]
    # plotting the values
    plt.plot(theta,outer)
    plt.plot(theta,outer_ex)
    plt.legend(['evaluated','exact'])

# plotting the values using the functions above
plt.figure()
plotinner(10)
plt.title('comparison inner')

plt.figure()
plotouter(10)
plt.title('comparison outer')

plt.show()
