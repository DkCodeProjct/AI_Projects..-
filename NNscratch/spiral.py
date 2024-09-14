import matplotlib.pyplot as plt

import numpy as np


# Didnt undestand that code NEED TO WORK ON THAT
def createData(points, classes):
    X = np.zeros((points * classes, 2))
    Y = np.zeros(points * classes, dtype='uint8')
    for classNum in range(classes):
        ix = range(classNum * points, (classNum + 1) * points)
        r = np.linspace(0.0, 1, points)
        t = np.linspace(classNum * 4, (classNum + 1) * 4, points) \
            + np.random.randn(points) * 0.2

        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        Y[ix] = classNum
    return X, Y

x, y = createData(100, 3)


# Scatter plot with color mapping
plt.scatter(x[:, 0], x[:, 1], c=y, cmap='brg')
plt.show()
