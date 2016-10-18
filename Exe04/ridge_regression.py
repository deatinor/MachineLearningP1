# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    w=np.linalg.solve(tx.T.dot(tx)+lamb**2*np.identity(tx.shape[1]),tx.T.dot(y))
    mse=sum((y-tx.dot(w))**2)/tx.shape[0]
    return mse,w
