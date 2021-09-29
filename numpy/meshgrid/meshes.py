#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt



X = np.array([1,2,3,4,5])

Y = np.array([11, 12, 13, 14, 15])

xx, yy = np.meshgrid(X,Y)

print("X coordinates:\n{}\n".format(xx))

print("Y coordinates:\n{}".format(yy))


#meshgrid with sparse=True
xx, yy = np.meshgrid(X,Y, sparse=True)

print("X coordinates:\n{}\n".format(xx))

print("Y coordinates:\n{}".format(yy))

#meshgrid of polar cordinates

R = np.linspace(1,5,10)

THETA = np.linspace(0, np.pi, 45)

radii, thetas = np.meshgrid(R,THETA)

print("R:{}".format(R.shape))

print("THETA:{}".format(THETA.shape))

print("meshgrid radii:{}".format(radii.shape))

print("mehgrid thetas:{}".format(thetas.shape))

#meshgrid with matrix indexing
i = np.array([1,2,3,4,5]) #rows
 
j = np.array([11, 12, 13, 14, 15]) #columns
 
ii, jj = np.meshgrid(i,j, indexing='ij')
 
print("row indices:\n{}\n".format(ii))
 
print("column indices:\n{}".format(jj))


#Flip the meshgrid

X = np.linspace(1,15,15)

Y = np.linspace(20,30,10)

xx, yy = np.meshgrid(X,Y)

fig = plt.figure()

ax = fig.add_subplot(111)

ax.plot(xx, yy, ls="None", marker=".")

plt.show()

#meshgrid with pyplot


ax = plt.subplot(111, polar=True)

ax.plot(thetas, radii, marker='.', ls='none')

plt.show()


