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



def Perceptron(X,y,eta, n_iter):
    width = int(X.size/len(X))
    w=[]
    for i in range(0,width+1):
        w.append(random()/10)
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
            updates.append(w)
            break
        n +=1
    
    updates = np.array(updates)
    return updates
        
percep = Perceptron(X,y,.1,10)

w = percep[-1]
updates = percep[:-1].tolist()
updates = np.array(updates)

x_1 = np.array([min(X[:,0]) - .25,max(X[:,0]) + .25])
x_2 = -(w[0]+w[1]*x_1)/w[2]




# plot data
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.plot(x_1,x_2)
# plt.savefig('images/02_06.png', dpi=300)
plt.show()

plt.plot(updates[:,0], updates[:,1],'*-')
plt.xlabel('Epochs')
plt.ylabel('Updates')    

#Currently, the program will return the values for w and the number of 
#updates for each epoch for any length of features, 
#but my plotting/solving doesn't work for that. I'll keep working on it