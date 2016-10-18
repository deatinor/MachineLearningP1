# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio: TODO
    # ***************************************************
    train_elements=int(ratio*x.shape[0])
    test_elements=x.shape[0]-train_elements
    indices = np.random.permutation(x.shape[0])
    training_idx, test_idx = indices[:train_elements], indices[train_elements:]
    x_train, x_test = x[training_idx], x[test_idx]
    y_train, y_test = y[training_idx], y[test_idx]
    return x_train,x_test,y_train,y_test
