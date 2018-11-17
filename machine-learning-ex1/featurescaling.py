import pandas as pd
import numpy as np
def feature(X,x):
    x = (x - X.mean())/X.std()
    return (x);
def gdMulti(X,Y,theta,alpha,iters):
    J = np.zeros(iters)
    m = len(Y)
    j=0
    while(iters!=0):
        hypo = np.dot(X,theta.T);
        sub = hypo - Y
        while(j!=X.shape[1]):
            ins = np.multiply(X[:,j],sub)
            theta[0,j] = theta[0,j] - alpha*np.sum(ins)/m
            j=j+1
        iters = iters-1;
    J = computeCost(X,Y,theta,m)
    return theta,J
def gd(X,Y,theta,alpha,num_iters):
    J_history  = np.zeros((num_iters,1))
    iter =0;
    m = len(Y)
    """print(X.shape[1])#3
    print(X.shape)#47 x 3
    print(theta.shape)# 1X3"""
    j=0
    while(iter<=num_iters):
        h = np.dot(X,theta.T)
        while(j<X.shape[1]):
            sol = np.multiply(X[:,j],(h-Y))#47x1 * 47x1
            theta[0,j] = theta[0,j] - alpha*(np.sum(sol))/m
            j=j+1
        iter = iter+1
    J_history=computeCost(X, Y, theta, m)
    return (theta,J_history)
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = 3
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))

        theta = temp

        cost[i] = computeCost(X, y, theta,len(y))

    return theta, cost
def computeCost(X,Y,theta,m):
    h = np.dot(X,theta.T)
    sol1 = h-Y
    sol1 = np.square(sol1)
    sol2 = np.sum(sol1)
    return(sol2/(2*m))
