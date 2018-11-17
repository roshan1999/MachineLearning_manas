import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from featurescaling import *
from sklearn import linear_model

#importing Data into X and Y
def main():
    data = pd.read_csv('ex1data2.txt',sep = ',',header = None)
    X = data.iloc[:,0:2]
    Y = data.iloc[:,2:3]
    X1 = X
    m=len(Y)
    print("Printing data: ")
    print(X.head())
    print(Y.head())
    #input("Press enter to continue");
    print('Normalizing data')
    X = feature(X1,X);
    X.insert(0,'',1);
    X.columns = [0,1,2]
    print(X.head())
    #input('Press enter to continue')
    print('Running Gradient descent --------------')
    alpha = 0.01
    num_iters = 1500
    X = np.array(X)
    Y = np.array(Y)
    theta = np.zeros((1,3))
    #print(theta)
    """theta, J_history = gdMulti(X, Y, theta, alpha, num_iters);
    print(theta)
    print(J_history)
    theta,J_history = gd(X,Y,theta,alpha,num_iters)
    print(theta)
    print(J_history)
    X = np.matrix(X)
    Y = np.matrix(Y)"""
    theta = np.matrix(np.zeros((1,3)))
    theta,J_history = gradientDescent(X, Y, theta, alpha, num_iters)
    #input('Press enter to ')
    print(theta)
    print(J_history)
    print('Computing cst- -----')
    J = computeCost(X,Y,theta,m)
    print(J)
    d = [1650,3]
    d = feature(X1,d);
    d = np.array(d)
    d = np.insert(d,0,1)
    print(d)
    #d.columns = [0,1,2]
    price = np.dot(d,theta.T)
    print("The price is:")
    print(price)
    plt.plot(np.arange(1,1501),J_history,'-b',linewidth =1)
    plt.show()
    """
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(np.arange(1, J_history, 'r'))
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')"""
if __name__ == '__main__':
    main()
