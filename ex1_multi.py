import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from featurescaling import *
#importing Data into X and Y
def main():
    data = pd.read_csv('ex1data2.txt',sep = ',',header = None)
    X = data.iloc[:,0:2]
    Y = data.iloc[:,2]
    m = len(Y)
    print("Printing data: ")
    print(X.head())
    input("Press enter to continue");
    print('Normalizing data')
    X,mu,sigma = feature(X);
    X.insert(0,'',1);
    X.columns = [0,1,2]
    print(X.head())
    input('Press enter to continue')
    print('Running Gradient descent --------------')
    alpha = 0.01
    num_iters = 400
    theta = np.zeros((3,1))
    y=Y
    Y = Y[:,np.newaxis]
    #print(theta)
    theta, J_history = gdMulti(X, Y, theta, alpha, num_iters);
    print(theta)
    input('Press enter to continue')
    print('Computing c0st- -----')
    J = computeCost(X,Y,theta,m)
    print(J)
    

if __name__ == '__main__':
    main()
