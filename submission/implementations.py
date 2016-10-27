# -*- coding: utf-8 -*-
u"""Our implementations of the regression functions.

Made by group #37:
 - Stefano Savarè
 - Camila González Williamson
 - Robin Weiskopf
"""

import numpy as np

def compute_loss(y, tx, w):
    """Calculates the loss with MSE."""
    N = y.shape[0]
    e = y - tx @ w
    loss = e.T @ e / (2 * N)
    return loss



def compute_gradient(y, tx, w):
    """Calculates the gradient on a """
    N = y.shape[0]
    e = y - tx @ w
    gradient = - (tx.T @ e) / N
    return gradient



def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent using MSE."""
    # Initialize dynamic point to initial_w
    w = np.copy(initial_w)

    for n_iter in range(max_iters):
        # compute gradient
        g = compute_gradient(y, tx, w)
        # update w by gradient
        w -= gamma * g

    # compute loss
    loss = compute_loss(y, tx, w)

    return w, loss



def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm using MSE."""
    return least_squares_minibatch(y, tx, initial_w, 1, max_iters, gamma)

def least_squares_minibatch(y, tx, initial_w, batch_size, max_iters, gamma):
    """Mini-batch stochastic gradient descent using MSE."""
    # Initialize dynamic point to initial_w
    w = np.copy(initial_w)

    # Construct the shuffled mini-batches
    ys = []
    txs = []
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        ys.append(minibatch_y)
        txs.append(minibatch_tx)

    for n_iter in range(max_iters):
        # select batch from array
        # (and prevent out of bounds errors)
        n = n_iter % int(y.shape[0] / batch_size)
        batch_y = ys[n]
        batch_tx = txs[n]
        # compute gradient
        g = compute_gradient(batch_y, batch_tx, w)

        # update w
        w -= gamma * g

    # compute loss
    loss = compute_loss(y, tx, w)

    return w, loss



def least_squares(y, tx):
    """Returns the optimal weights and the loss"""
    w = np.linalg.inv(tx.T.dot(tx)).dot(tx.T).dot(y)
    loss = compute_loss(y, tx, w)
    return w, loss



def build_poly(x, degree):
    """Constructs and returns a polynomial basis functions
    for input data x, for j=0 up to j=degree."""
    output = []
    for i in range(degree+1):
        x_i = x ** i
        output.append(x_i)

    return np.array(output).T



def split_data(x, y, ratio, seed=1):
    """Splits the dataset based on the ratio and returns a tuple of size 4
    containing: (x_train, x_test, y_train, y_test)"""
    # Calculate size of the
    N = x.shape[0]
    train_size = int(ratio * N)

    # Create index permutation and use it on x and y
    indices = np.random.permutation(N)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    # Separate the arrays
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return x_train, x_test, y_train, y_test



def ridge_regression(y, tx, lambda_):
    """Calculates optimal weights using ridge regression."""
    N = y.shape[0]
    w = np.linalg.inv(tx.T.dot(tx) - 2 * N * lamb * np.ones(tx.shape[1])).dot(tx.T).dot(y)
    loss = compute_loss(y, tx, w)
    return w, loss
