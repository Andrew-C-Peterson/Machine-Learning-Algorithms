#A one vs all Logistic Regression model

#looking at the Iris data set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

start = time.time()


#Here I import the data set and split into X and y
df = pd.read_csv('iris.data', header=None)

# select setosa and versicolor
y = df.iloc[0:150, 4].values

# extract sepal length and petal length
X = df.iloc[0:150, 0:4].values

class LogRova(object):
    """This performs a Logistic Regression using gradient descent.
    Using the oneVall function it can handle multiple classes"""
    
    def __init__(self, n_iter,eta, random_state = None):
        """Initialize values for some of the variables"""
        
        self.n_iter = n_iter
        self.eta = eta
        self.cat = np.unique(y)
        self.num_cat = len(self.cat)
        self.seed = random_state
        
           
    def fit(self,X,y,j):
        """Fits a logistic regression model to data. This only works for 
        binary classification"""
        
        width = int(X.size/len(X))
        w=[]
        for i in range(0,width+1):
            if self.seed == None:
                w.append(random.random()/100)
            else:
                random.seed(a=self.seed*3)
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
        """This function can be used to fit a model to
                        handle more than two classes."""
        
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
        """This funciton predicts values for data based on the fit model"""
        
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
        """Returns the accuracy from the predictions"""
        
        count = 0
        for i in range(0,len(y_test)):
            if self.y_hat_test[i] == y_test[i]:
                count +=1
        
        self.acc_test = 100*count/len(y_test)
        return self
    
    def conf_mat(self, y, y_hat):
        """Returns a confusion matrix given the actual and predicted values"""
        
        conf_mat = np.zeros((self.num_cat,self.num_cat))
        
        for i in range(0,len(y)):
            pred = np.where(self.cat == y_hat[i])[0][0]
            act = np.where(self.cat == y[i])[0][0]
            conf_mat[pred,act]+=1
            
        rows = []
        col = []
        
        for i in self.cat:
            rows.append('Predicted ' + str(i))
            col.append('Actual ' + str(i))
    
        self.conf = pd.DataFrame(conf_mat, index = rows, columns = col)
        
        
        return self
            

#Split into the train and test data sets     
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8,
                                                random_state = 1)

#I use a min max scaler to scale the features. This helps increase
#accuracy a lot.
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Here I initialize the Log Reg model and use the one vs all method to train
LR1 = LogRova(n_iter=1000, eta = .05, random_state  = 1).oneVall(X_train,y_train)
acc_train = LR1.acc
confusion_matrix_train = LR1.conf_mat(y_train, LR1.y_hat).conf

#Now I make predictions using the test data
LR1.predict(X_test).test_acc(y_test)
acc_test = LR1.acc_test
confusion_matrix_test = LR1.conf_mat(y_test, LR1.y_hat_test).conf


print('Train accuracy is: ', acc_train)
print('Test accuracy is: ', acc_test)
print('Total time is: ', time.time()-start)

