#!/usr/bin/env python
# coding: utf-8

# In[759]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[760]:


data = pd.read_csv("../datasets/Fish.csv")


# In[761]:


data.groupby(["Species"]).describe()

print(data)


# In[762]:


def Sigmoid(z):
    return 1/(1+np.exp(-z))


# In[763]:


def h(x,theta):
    return Sigmoid(x.dot(theta))


# In[764]:


def J(X,y,theta,m):
    hx = h(X,theta)
    return (-y.transpose().dot(np.log(hx))-(1-y.transpose()).dot(np.log(1-hx)))/m


# In[765]:


def Gradient(X,theta,lr,m):
    return theta - lr*X.transpose().dot(h(X,theta)-y)/m


# In[766]:


dic = {"Bream":0,"Perch":1}

breams = data.loc[ data["Species"] == "Bream"]
perchs = data.loc[ data["Species"] == "Perch"]

bm = breams.mean().values
pm = perchs.mean().values
bstd = breams.std().values
pstd = perchs.std().values

bv = breams.drop(["Species"],axis=1)
bv = (bv-bm)/bstd

pv = perchs.drop(["Species"],axis=1)
pv = (pv-m)/pstd

bv = pd.DataFrame(bv)
pv = pd.DataFrame(pv)


data_bp =  breams.append(perchs)
X =  bv.append(pv)
breams = data_bp.insert(1,"",1)

print(data_bp)

lr = 0.03

y = data_bp["Species"]

print(y)

y = y.replace(dic).values


print(X,X.shape)

X = X.values

n = X.shape[1]
m = y.size

EPOCH = 100000

theta = np.zeros(n)


# In[767]:


for i in range(EPOCH):
    theta = Gradient(X,theta,lr,m)
    print(J(X,y,theta,m))


# In[768]:


h(X[0],theta) < 0.5


# In[769]:


ys = []
yp = []

for i in range(m):
    if(h(X[i],theta) < 0.5):
        ys.append(0)
    else:
        ys.append(1)
        
for i in range(m):
    yp.append(h(X[i],theta))


# In[770]:


x = range(m)


plt.scatter(x,ys,s=5)
plt.scatter(x,yp,s=5)
plt.title('Bream or Perch predictions')
plt.xlabel('x')
plt.ylabel('y')

plt.show()

plt.scatter(x,y,s=5)
plt.title('Bream or Perch real')
plt.xlabel('x')
plt.ylabel('y')

plt.show()

