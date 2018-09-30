import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lineardef import *

def main():
    data = pd.read_csv('ex1data1.txt',header = None, names = ['Population','Profits'])
    #data.head() #Prints fiew values from dataframe
    #data.describe(); #gives description on the few values
    data.plot(kind =    "scatter",x = 'Population', y = 'Profits')
    data.insert(0,'ones',1)
    cols = data.shape[1]
    print(data.shape)
    X = data.iloc[:,0:2];
    Y = data.iloc[:,2:3];#DOUBT: WHY DOESNT Y = data.iloc[:,2] work????????!!!!!!!!!!!!!!!!!!!
    print(X);print(Y)
    X = np.array(X)
    Y = np.array(Y)
    theta = np.zeros((1,2))
    print(theta.shape)
    J = computecost(X,Y,theta)
    print("Cost function without optimization: ",J)
    alpha = 0.01
    iters = 1000
    theta_upd,J = gradientDescent(X,Y,alpha,iters,theta)
    print("Cost function with optimization:",J)
    input("Press enter to See the plot")
    x = np.linspace(data.Population.min(),data.Population.max(),100)#x = np.linspace(start,stop,number of points)
    #print(x)
    x= np.reshape(x,(100,1))
    #print(x)
    f = np.dot(x,theta_upd)
    fig,ax = plt.subplots()
    ax.plot(x,f,'r',label = 'Predicted Values')
    ax.scatter(data.Population,data.Profits,label = 'Training Data')
    ax.legend(loc =2)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Graph of Population vs Profit')
    plt.show();
if __name__ == '__main__':
    main()
