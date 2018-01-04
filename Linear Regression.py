"""
@author: Shubham Shantaram Pawar
"""

import pandas as pd
import numpy as np
import matplotlib .pyplot as plt
    
def computeCost(X, y, theta):
    m = len(y)
    h = np.matmul(X, theta)
    J = (1/(2 * m)) * np.sum(np.square(np.subtract(h,y)))
    return J

def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros((num_iters,1))
    
    for iter in range(0, num_iters):
        h = np.matmul(X, theta)
        
        theta = np.subtract(theta, ((alpha/m) * np.matmul(X.T, np.subtract(h, y))))
        
        J_history[iter] = computeCost(X, y, theta)
        
    return (theta, J_history)

def main():
    df = pd.read_csv(
    filepath_or_buffer='linear_regression_test_data.csv', 
    header=0, 
    sep=',')

    df.dropna(how="all", inplace=True)
    df.tail()

    data = df.iloc[0:,1:].values

    x = data[:,0]
    y = data[:,1]
    y_theoretical = data[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Original Data')
    ax.scatter(x, y, color='blue')
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.set_size_inches(20, 12)
    fig.show()
    
    m = len(x)
    x = np.reshape(x, (m,1))
    y = np.reshape(y, (m,1))
    
    theta = np.zeros((2,1))
    
    iterations = 500
    alpha = 0.1
    
    X = np.concatenate((np.ones((m,1)),x),axis=1)
    
    theta, J_history = gradientDescent(X, y, theta, alpha, iterations)
    
    print("theta 0:", theta[0][0])
    print("theta 1:", theta[1][0])
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Regression Line')
    ax.scatter(x, y, color='blue', label = 'y vs x')
    ax.scatter(x, y_theoretical, color='red', label = 'y-theoretical vs x')
    plt.plot(x, theta[0][0] + x * theta[1][0], 'r-', label = 'Regression Line')
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x', fontsize = 13)
    ax.set_ylabel('y / y-theoretical', fontsize = 13)
    ax.legend()
    fig.set_size_inches(20, 15)
    fig.show()
    plt.savefig('Regression Line')
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(r'iteration')
    ax.set_ylabel(r'$J(\theta)$')
    ax.scatter(range(iterations), J_history, color='blue', s=10)
    fig.set_size_inches(8, 5)
    fig.show()
    
if __name__ == '__main__':
    main()
