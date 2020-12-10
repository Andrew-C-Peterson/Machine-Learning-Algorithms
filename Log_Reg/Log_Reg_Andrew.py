#This is a simple logistic regression binary classifier. 
#Giving it the training data and it will output the array 'w'
#which is the weight for each feature.

#I do this on a sample from the iris data set (with only 2 features)
#and then plot


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import random
import math
import time

start = time.time()


#import the data
df = pd.read_csv('iris.data', header=None)

# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values


#Here is my class for the logistic regression
class LogR(object):
    
    def __init__(self, n_iter,eta):
        #Initializes. I could give it standard values for these variables
        #but I leave it so you have to fill them in
        self.n_iter = n_iter
        self.eta = eta
        
           
    def fit(self,X,y):
        #I found the width in a bit of weird way..Maybe an easier command for this
        width = int(X.size/len(X))
        #Here I initialize the weights, 'w'
        w=[]
        #I want to have small random numbers to start.
        #Again, maybe a better way to do this
        for i in range(0,width+1):
            w.append(random()/10)
            
        #n counts the number of iterations, error_epoch stores the error for 
        #each epoch, and N stores the number for each iteration
        n = 1
        error_epoch=[]
        N=[]
        
        while n < self.n_iter+1:
            #For each epoch, the cost and error start as 0
            cost_total = [0,0,0]
            error_sum_sq = 0
            
            #For each value, I estimate y as phi, then calculate the 
            #'cost' and the error
            for i in range(0, len(y)):
                x = np.array([1, X[i,0],X[i,1]])
                z = np.dot(w,x)
                phi = 1/(1+math.exp(-z))
                cost = (y[i]-phi)*x
                
                #Error and cost are summed over the whole epoch
                error_sum_sq+=(y[i]-phi)**2
                cost_total +=cost
            
            #Final error times (1/2), update w, and move on
            error_sum_sq = error_sum_sq*.5
            error_epoch.append(error_sum_sq)
            w += cost_total*self.eta
            N.append(n)
            
            n+=1
        
        self.error=np.array(error_epoch)
        self.w = w
        self.N = N
        return self

#In the following code, I fit the logistic regression to the data
#and plot
   
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