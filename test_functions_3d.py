import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random


def fun(x, y):
    # return x * y
    # return 2*np.power((x+y)/2,2) - 2*(x+y)/2 +1
    # return ((np.power((x + y) / 2, 2) - (x + y) / 2 + -1/4) + 0.5)*5
    return (((np.power((x + y) / 2, 2) - (x + y) / 2 + -1/4))+0.5)*4




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(0, 1.0, 0.05)
# g = np.arange([0, 0.5, 1])
# t = np.arange([0, 0.5, 1])

X, Y = np.meshgrid(x, y)
zs = np.array([fun(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

ax.plot_surface(X, Y, Z)

ax.set_xlabel('Gradient')
ax.set_ylabel('Time')
ax.set_zlabel('Chance to Drop')

plt.show()
print("finished")