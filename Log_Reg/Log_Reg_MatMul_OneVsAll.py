#A one vs all Logistic Regression model. I make a random dataset 
#to test this on, since it needs to be linearly seperable.
#It works with as many features as you want, but I practiced with just 2
#for better visualization


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
from sklearn.model_selection import train_test_split

start = time.time()

#For the following code, I am just making my random data set
X = []
y =[]

for i in range(0,50):
    X.append([random.random(),random.random()])
    y.append('a')

for i in range(0,50):
    X.append([random.random()+ 1.5, random.random()])
    y.append('b')
    
for i in range(0,50):
    X.append([random.random()+ .75, random.random()+2])
    y.append('c')    

X = np.array(X)
y = np.array(y)

plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='a')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='b')
plt.scatter(X[100:150, 0], X[100:150, 1],
            color='green', marker='^', label='c')

class LogRova(object):
    
    def __init__(self, n_iter,eta, random_state = None):
        self.n_iter = n_iter
        self.eta = eta
        self.cat = np.unique(y)
        self.num_cat = len(self.cat)
        self.seed = random_state
        
           
    def fit(self,X,y,j):
        width = int(X.size/len(X))
        w=[]
        for i in range(0,width+1):
            random.seed(a=self.seed)
            w.append(random.random()/100)
        
        X_mat = np.concatenate((np.ones(len(y))[:, np.newaxis], X), axis=1)
        n = 1
        error_epoch=[]
        N=[]
        
        while n < self.n_iter+1:
                           
            z = np.matmul(X_mat,w)
            phi = 1/(1+np.exp(-z))
            cost_total = np.dot((y-phi).T,X_mat)
            
            error_sq = (y-phi)**2
            
        
            error_sq = error_sq*.5
            error_epoch.append(sum(error_sq))
            w += cost_total*self.eta
            N.append(n)
            
            n+=1
        
        self.phi = phi
        self.error=np.array(error_epoch)
        self.w = w
        self.N = N
        
        return self
        
    def oneVall(self,X,y):
        
        w_array = []
        error_array = []
        phi_array = np.zeros((len(y),self.num_cat))
        
        for j in range(0, self.num_cat):
            y_temp = np.where(y == self.cat[j], 1, 0)
            LR1 = LogRova.fit(self, X, y_temp,j)
            
            w_array.append(LR1.w)
            error_array.append(LR1.error)
            phi_array[:,j]=LR1.phi
        
        self.w_array = np.array(w_array)
        self.error_array = np.array(error_array)
        self.phi_array = phi_array.argmax(axis =1)
        
        y_hat = []
        count = 0
        for i in range(0,len(y)):
            y_hat.append(self.cat[self.phi_array[i]])
            if y_hat[i] == y[i]:
                count +=1
        
        self.y_hat = np.array(y_hat)
        self.acc = 100*count/len(y)
        return self
    
    def predict(self,X):
        X_mat_test = np.concatenate((np.ones(len(X[:,0]))[:, np.newaxis], X), axis=1)
        phi_test = np.zeros((len(X[:,0]),self.num_cat))
        
        for j in range(0, self.num_cat):
            z = np.matmul(X_mat_test,self.w_array[j])
            phi_test[:,j] = 1/(1+np.exp(-z))
        
        
        self.phi_test = phi_test.argmax(axis = 1)
        y_hat = []
        for i in range(0,len(X[:,0])):
            y_hat.append(self.cat[self.phi_test[i]])
            
        self.y_hat_test = np.array(y_hat)
        return self
    
    def test_acc(self,y_test):
        count = 0
        for i in range(0,len(y_test)):
            if self.y_hat_test[i] == y_test[i]:
                count +=1
        
        self.acc_test = 100*count/len(y_test)
        return self
        
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8,
                                                    random_state = 1)

LR1 = LogRova(n_iter=1000, eta = .05).oneVall(X_train,y_train)

w = LR1.w_array
acc_train = LR1.acc

x_1 = np.array([0,2.5])
x_2 = (0.5-w[0,0]-w[0,1]*x_1)/w[0,2]  
plt.plot(x_1,x_2,'r')

x_2 = (0.5-w[1,0]-w[1,1]*x_1)/w[1,2]  
plt.plot(x_1,x_2,'b')

x_2 = (0.5-w[2,0]-w[2,1]*x_1)/w[2,2]  
plt.plot(x_1,x_2,'g')

plt.ylim([-1,4])
plt.figure()

LR1.predict(X_test)
LR1.test_acc(y_test)
acc_test = LR1.acc_test

print('Train accuracy is: ', acc_train)
print('Test accuracy is: ', acc_test)
print('Total time is: ', time.time()-start)