import numpy as np
def computecost(X,Y,theta):
    m = len(Y)
    h = np.dot(X, theta.T);
    sumof = np.power((h - Y),2)
    sum = np.sum(sumof)/(2 * m)
    return sum
def gradientDescent(X,Y,alpha,iters,theta):
    J = np.zeros(iters)
    m = len(Y)
    while(iters!=0):
        hypo = np.dot(X,theta.T);
        sub = hypo - Y
        ins = np.dot(X[:,0],sub)
        theta[:,0] = theta[:,0] - alpha*ins/m
        ins = np.dot(X[:,1],sub)
        theta[:,1] = theta[:,1] - alpha*ins/m
        iters = iters-1;
    J = computecost(X,Y,theta)
    return theta,J;
