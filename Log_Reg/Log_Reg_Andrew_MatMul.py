#This is a simple logistic regression binary classifier. 
#Giving it the training data and it will output the array 'w'
#which is the weight for each feature.

#I do this on a sample from the iris data set (with only 2 features)
#and then plot



#I use matrix multiplication here instead of looping, which makes the code
#much more efficient. Yay! The rest of the code is the same, 
#so no comments

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import random
import time

start = time.time()

df = pd.read_csv('iris.data', header=None)

# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values


class LogR(object):
    
    def __init__(self, n_iter,eta):
        self.n_iter = n_iter
        self.eta = eta
        
           
    def fit(self,X,y):
        width = int(X.size/len(X))
        w=[]
        for i in range(0,width+1):
            w.append(random()/10)
        
        n = 1
        error_epoch=[]
        N=[]
        X = np.concatenate((np.ones(len(y))[:, np.newaxis], X), axis=1)
        
        while n < self.n_iter+1:
                           
            z = np.matmul(X,w)
            phi = 1/(1+np.exp(-z))
            cost_total = np.dot((y-phi).T,X)
            
            error_sq = (y-phi)**2
            

            error_sq = error_sq*.5
            error_epoch.append(sum(error_sq))
            w += cost_total*self.eta
            N.append(n)
            
            n+=1
        
        self.error=np.array(error_epoch)
        self.w = w
        self.N = N
        return self
    
LR1 = LogR(n_iter=100, eta = .05).fit(X, y)
w = LR1.w

plt.plot(LR1.N,LR1.error)
plt.figure()

x_1 = np.array([4,7])
x_2 = (0.5-w[0]-w[1]*x_1)/w[2]
    
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
    

plt.plot(x_1,x_2, label = 'n_iter = 1000, eta = 0.05')
plt.legend()

print(time.time()-start)