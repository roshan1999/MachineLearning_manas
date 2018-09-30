import pandas as pd
import numpy as np
def feature(X):
    mean = np.mean(X);
    std = np.std(X);
    X = (X - mean) /std
    return (X,mean,std);
def gdMulti(X,Y,theta,alpha,num_iters):
    m = len(Y);
    J_history  = np.zeros((num_iters,1))
    iter =1;

    while(iter!=num_iters):
        h = np.dot(X,theta)
        sol = np.dot(X.T,h-Y)
        theta = theta - alpha*sol/m
        iter=iter+1
    return (theta,J_history)
def computeCost(X,Y,theta,m):
    h = np.dot(X,theta)
    sol1 = h-Y
    sol1 = np.square(sol1)
    sol2 = np.sum(sol1)
    return(sol2/(2*m))
