#!/usr/bin/env python
import assignment2 as a2
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sps

# Maximum number of iterations.  Continue until this limit, or when error change is below tol.
max_iter = 500
tol = 0.00001

# Step size for gradient descent.
eetas = [0.5, 0.3, 0.1, 0.05, 0.01]

# Load data.
data = np.genfromtxt('data.txt')

# Data matrix, with column of ones at end.
X = data[:, 0:3]
# Target values, 0 for class 1, 1 for class 2.
t = data[:, 3]
# For plotting data
class1 = np.where(t == 0)
X1 = X[class1]
class2 = np.where(t == 1)
X2 = X[class2]

DATA_FIG = 1
epos = 0.000000001

for eeta in eetas:
    # Initialize w.
    w = np.array([0.1, 0, 0])
    size = X.shape[0]
    ind = (range(size))
    # Error values over all iterations.
    e_all = []
    for iter in range(0, max_iter):
        e = 0
        np.random.shuffle(data)
        for i in range(size):
            # Compute output using current w on all data X.
            y = sps.expit(np.dot(X[i], w))

            # e is the error, negative log-likelihood (Eqn 4.90)
            e += -(np.multiply(t[i], np.log(epos + y)) + np.multiply((1 - t[i]), np.log(epos + 1 - y)))

            # Gradient of the error, using Eqn 4.91
            grad_e = (np.multiply((y - t[i]), X[i].T))

            # Update w, *subtracting* a step in the error derivative since we're minimizing
            #             w_old = w
            w = w - eeta * grad_e

        e = e / size
        e_all.append(e)

        print("epoch {0:d}, negative log-likelihood {1:.4f}, w={2}".format(iter, e, w.T))

        if iter > 0:
            if np.absolute(e - e_all[iter - 1]) < tol:
                break

    plt.plot(e_all, label=str(eeta) + ' ,Learning Rates')
    plt.legend()
    plt.ylabel('Negative log likelihood')
    plt.title('Training logistic regression Stochastic Gradient Descent')
    plt.xlabel('Epoch')
plt.show()
