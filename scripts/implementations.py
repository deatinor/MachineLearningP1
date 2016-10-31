# -*- coding: utf-8 -*-
u"""Our implementations of the regression functions.

Made by group #37:
 - Stefano Savarè
 - Camila González Williamson
 - Robin Weiskopf
"""



import numpy as np

from helpers import batch_iter




# ***********************************
#        Utility functions
# ***********************************

def compute_loss(y, tx, w):
    """Calculates the loss with MSE."""
    N = y.shape[0]
    e = y - tx @ w
    loss = e.T @ e / (2 * N)
    return loss



def compute_gradient(y, tx, w):
    """Calculates the gradient with MSE."""
    N = y.shape[0]
    e = y - tx @ w
    gradient = - (tx.T @ e) / N
    return gradient



def compute_stoch_gradient(y, tx, w):
    """Calculates the gradient of SGD using MSE."""
    N = y.shape[0]
    e = y - tx @ w	
    gradient = (tx.T @ e) / (-N)
    return gradient


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
    print(indices)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    # Separate the arrays
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return x_train, x_test, y_train, y_test


def evaluate(y,tX,w):
    prediction=compute_predictions(tX,w)
    return evaluate_prediction(prediction,y)

def evaluate_prediction(prediction,y):
    return (sum(y*prediction)/y.shape[0]+1)/2

def compute_predictions(tX,w):
    prediction=tX.dot(w)
    prediction[np.where(prediction <= 0)] = -1
    prediction[np.where(prediction > 0)] = 1
    return prediction


# ***********************************
#       Regression functions
# ***********************************

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent"""
    # Initialize dynamic point to initial_w
    w = np.copy(initial_w)

    for n_iter in range(max_iters):
        # compute gradient
        g = compute_gradient(y, tx, w)
        # update w by gradient
        w -= gamma * g
        if n_iter%1000==0:
            print(n_iter)

    # compute loss
    loss = compute_loss(y, tx, w)

    return w, loss



def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent algorithm"""
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
        g = compute_stoch_gradient(batch_y, batch_tx, w)

        # update w
        w -= gamma * g

    # compute loss
    loss = compute_loss(y, tx, w)

    return w, loss



def least_squares(y, tx):
    """Least Squares regression using the normal equiations
        Returns the optimal weights and the loss"""
    w = np.linalg.solve(tx.T.dot(tx),tx.T.dot(y))
    loss = compute_loss(y, tx, w)
    return w, loss



def ridge_regression(y, tx, lambda_):
    """Calculates optimal weights using ridge regression with normal equations"""
    N = y.shape[0]
    w = np.linalg.solve(tx.T.dot(tx) + (lambda_**2) * np.identity(tx.shape[1]), tx.T.dot(y))
    loss = compute_loss(y, tx, w)
    return w, loss



def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic Regression using gradient descent"""
    
    #Init parameters
    threshold = 1e-8
    losses = []
    w = np.copy(initial_w)
    
    # start the logistic regression
    for iter in range(max_iters):
        
        #Calculate actual loss and gradient
        loss = calculate_loss_logit(y, tx, w)
        grad = calculate_gradient_logit(y, tx, w)
        
        #Update w
        w = w - gamma*grad
        
        # converge criteria
        losses.append(loss)

        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, losses[-1]


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent"""
   
    #Init parameters
    threshold = 1e-8
    losses = []
    w = np.copy(initial_w)
    
    # start the logistic regression
    for iter in range(max_iters):
        
        #Calculate actual loss and gradient

        loss = calculate_loss_logit(y, tx, w) + lambda_*np.sum(w*w)
        grad = calculate_gradient_logit(y, tx, w) + 2*lambda_*w
        
        #Update w
        w = w - gamma*grad
        
        # converge criteria
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, losses[-1]

def reg_logistic_regression_newton(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent"""
   
    #Init parameters
    threshold = 1e-8
    losses = []
    w = np.copy(initial_w)
    
    # start the logistic regression
    for iter in range(max_iters):
        #Calculate actual loss and gradient
        loss = calculate_loss_logit(y, tx, w) + lambda_*np.sum(w*w)
        grad = calculate_gradient_logit(y, tx, w) + 2*lambda_*w
        H = calculate_hessian_logit(y, tx, w) + 2*lambda_*np.identity(w.shape[0])
        #Update w
        w = w - gamma*np.linalg.inv(H).dot(grad)
        
        
        # converge criteria
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, losses[-1]


def sigmoid(t):
    """applies sigmoid function on t."""
    t[t<-30]=-30
    result = 1/(1+np.exp(-t))
    return result

def calculate_hessian_logit(y, tx, w):
    """Returns the hessian of the loss function"""
    sig = sigmoid(tx.dot(w))

    #S = np.identity(tx.shape[0])*(sig*(1-sig))
    S = sig*(1-sig) 
    H = ((tx.T)*S).dot(tx)
    return H

def calculate_gradient_logit(y, tx, w):
    """compute the gradient of loss."""
    grad = (tx.T).dot(sigmoid(tx.dot(w))-y)
    return grad

def calculate_loss_logit(y, tx, w):
    """compute the cost by negative log likelihood."""
    dot_product=tx.dot(w) 
    log=np.zeros(dot_product.shape[0])
    log[dot_product>40]=40
    log[dot_product<=40] = np.log(1+np.exp(dot_product[dot_product<=40]))
    loss = np.ones(y.shape[0]).dot(log) - (y.T.dot(tx)).dot(w)
    return loss





