import numpy as np
def computecost(X,Y,theta):
    m = len(Y)
    h = np.dot(X, theta.T);
    sumof = np.power((h - Y),2)
    sum = np.sum(sumof)/(2 * m)
    return sum
"""def gradientDescent(X,Y,alpha,iters,theta):
    J = np.zeros(iters)
    m = len(Y)
    for i in range(iters):
        hypo = np.dot(X,theta.T);
        sub = hypo - Y
        ins = np.dot(X[:,0],sub)
        theta[:,0] = theta[:,0] - alpha*np.sum(ins)/m
        ins = np.dot(X[:,1],sub)
        theta[:,1] = theta[:,1] - alpha*np.sum(ins)/m
        J[i] = computecost(X,Y,theta)
    return theta,J;"""
def gradientDescent(X, y, theta, alpha, iters):
        temp = np.matrix(np.zeros(theta.shape))
        parameters = 2
        cost = np.zeros(iters)

        for i in range(iters):
            error = (X * theta.T) - y

            for j in range(parameters):
                term = np.multiply(error, X[:,j])
                temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))

            theta = temp

            cost[i] = computeCost(X, y, theta,len(y))

        return theta, cost
