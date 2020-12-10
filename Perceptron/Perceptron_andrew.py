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

# plot data
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')




w = [random()/10, random()/10, random()/10]

eta = .1
n_iter = 10
n = 1
updates = []

while n < n_iter+1:
    updates.append([])
    error = 0
    for i in range(0, len(y)):
        x = [1, X[i,0],X[i,1]]
        z = np.dot(w,x)
        if z >=0:
            predict = 1
        else:
            predict = -1
        
        diff =  y[i]-predict
        
        if diff != 0:
            for j in range(0,len(w)):
                w[j] += eta*(diff)*x[j]
            error+=1
    
    updates[n-1].append(n)
    updates[n-1].append(error)
    if error == 0:
        break
    n +=1

updates = np.array(updates)

x_1 = np.array([4,7])
x_2 = -(w[0]+w[1]*x_1)/w[2]
plt.plot(x_1,x_2)



# plt.savefig('images/02_06.png', dpi=300)
plt.show()

plt.plot(updates[:,0], updates[:,1],'*-')
plt.xlabel('Epochs')
plt.ylabel('Updates')    



        