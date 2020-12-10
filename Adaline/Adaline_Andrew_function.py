import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import random

df = pd.read_csv('iris.data', header=None)

# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values


class AD(object):
    
    def __init__(self, eta, n_iter):
        self.eta = eta
        self.n_iter = n_iter
        
           
    def fit(self,X,y):
        width = int(X.size/len(X))
        w=[]
        for i in range(0,width+1):
            w.append(random()/10)
        n = 1
        error_epoch=[]
        N=[]
        
        while n < self.n_iter+1:
            cost_total = [0,0,0]
            error_sum_sq = 0
            for i in range(0, len(y)):
                x = np.array([1, X[i,0],X[i,1]])
                z = np.dot(w,x)
                cost = (y[i]-z)*x
                error_sum_sq+=(y[i]-z)**2
                cost_total += cost
            
            cost_total = cost_total*self.eta
            error_sum_sq = error_sum_sq*.5
            error_epoch.append(error_sum_sq)
            w += cost_total
            N.append(n)
            
            n+=1
        
        self.error=np.array(error_epoch)
        self.w = w
        self.N = N
        return self
    
ada1 = AD(n_iter=10, eta=0.001).fit(X, y)
w = ada1.w

plt.plot(ada1.N,ada1.error)
plt.title('n_iter = 10, eta = 0.001')
plt.figure()

x_1 = np.array([4,7])
x_2 = -(w[0]+w[1]*x_1)/w[2]


ada2 = AD(n_iter=50, eta = .0001).fit(X,y)
w_2 = ada2.w
plt.plot(ada2.N,ada2.error)
plt.title('n_iter = 50, eta = 0.0001')
plt.figure()
    
x_2_2 = -(w_2[0]+w_2[1]*x_1)/w_2[2]

ada3 = AD(n_iter=10, eta = .0001).fit(X,y)
w_3 = ada3.w
plt.plot(ada3.N,ada3.error)
plt.title('n_iter = 10, eta = 0.0001')
plt.figure()
    
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
    
x_2_3 = -(w_3[0]+w_3[1]*x_1)/w_3[2]
plt.plot(x_1,x_2_3, label = 'n_iter = 10, eta = 0.0001')
plt.plot(x_1,x_2_2, label = 'n_iter = 50, eta = 0.0001')
plt.plot(x_1,x_2, label = 'n_iter = 10, eta = 0.001')
plt.legend()
