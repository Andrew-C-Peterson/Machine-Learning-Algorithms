import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

df = pd.read_csv('iris.data', header=None)

# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values


def rand_order(y):
    rand_list=[]
    for i in range(0,len(y)):
        rand_list.append(i)
    random.shuffle(rand_list)
    return rand_list

w = [random.random()/10, random.random()/10, random.random()/10]

eta = .01
n_iter = 15
n = 1
error_epoch=[]
N=[]

while n < n_iter+1:
    error_sum_sq = 0
    rand_list = rand_order(y)
    
    for i in rand_list:
        x = np.array([1, X[i,0],X[i,1]])
        z = np.dot(w,x)
        cost = (y[i]-z)*x
        error_sum_sq+=(y[i]-z)**2
        
        w+=cost*eta
    
    error_sum_sq = error_sum_sq*.5
    error_epoch.append(error_sum_sq)
    N.append(n)
        
    n+=1

# plot data
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
    
x_1 = np.array([4,7])
x_2 = -(w[0]+w[1]*x_1)/w[2]
plt.plot(x_1,x_2)

plt.figure()

plt.plot(N,error_epoch,'*-')