#This is a simple logistic regression binary classifier. 
#Giving it the training data and it will output the array 'w'
#which is the weight for each feature.

#I do this on a sample from the iris data set (with only 2 features)
#and then plot

#For this code, I use stochastic gradient descent (SGD)
#ie, I update the weights after each value, not after the full epoch

#Thus, I can't do matmul and I need to randomize order. So I will
#comment the differences.

#The benefit is that this could converage faster


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
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
        
    def rand_order(y):
        #In this function, I make a list of randomly ordered indices
        #for a length of 'y'
        rand_list=[]
        for i in range(0,len(y)):
            rand_list.append(i)
        random.shuffle(rand_list)
        return rand_list
        
           
    def fit(self,X,y):
        width = int(X.size/len(X))
        self.w=[]
        for i in range(0,width+1):
            self.w.append(random.random()/10)
        n = 1
        self.error_epoch=[]
        self.N=[]
        
        while n < self.n_iter+1:
            
            error_sum_sq = 0
            #Here I create the random ordered list
            rand_list = LogR.rand_order(y)
            
            #Here I iterate throught the randomly ordered list
            for i in rand_list:
                #Calculate the 'cost' for each value
                #Then update the weights 'w'
                #Also update the error for the whole epoch
                x = np.array([1, X[i,0],X[i,1]])
                z = np.dot(self.w,x)
                phi = 1/(1+math.exp(-z))
                cost = (y[i]-phi)*x
                self.w += cost*self.eta              
                error_sum_sq+=(y[i]-phi)**2
                
    
            #Store the error for each epoch
            error_sum_sq = error_sum_sq*.5
            self.error_epoch.append(error_sum_sq)
            self.N.append(n)
            
            n+=1
        return self
    
LR1 = LogR(n_iter=100, eta = .05).fit(X, y)
w = LR1.w

plt.plot(LR1.N,LR1.error_epoch)
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