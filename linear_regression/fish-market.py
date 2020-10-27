#!/usr/bin/env python
# coding: utf-8

# In[862]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv


# In[863]:


df = pd.read_csv("../datasets/Fish.csv")


# In[864]:


df.groupby(["Species"]).count()


#"Perch"


# In[865]:


df_perch = df.loc[ df["Species"] == "Perch"]
df_perch = df_perch.drop(["Species"],axis=1)


# In[866]:


y = df_perch["Weight"].values
X = df_perch.iloc[:,1:].values

n = df_perch.iloc[:,2:].shape[1]+1
m = df_perch.iloc[1:].count()[0]

print(n)
print(m)

gamma = df_perch.iloc[:,1:].std().values
mu = df_perch.iloc[:,1:].mean().values
theta = np.zeros(n)
lr = 0.3
EPOCH = 15


# In[867]:


mu


# In[868]:


X = (X-mu)/gamma


# In[869]:


def h(x):
    return x.dot(theta)


# In[870]:


def J():
    A = (X.dot(theta)-y).transpose()
    B = (X.dot(theta)-y)
    return A.dot(B)/2*m


# In[871]:


def Gradient():
    return theta - lr*X.transpose().dot(X.dot(theta)-y)/m


# In[872]:


J_history = []
for i in range(EPOCH):
    theta = Gradient()
    J_history.append(J())


# In[873]:


fig, ax = plt.subplots()
t= np.arange(0,EPOCH,1)
ax.plot(t,J_history)
ax.set(xlabel='Epochs', ylabel='Cost function',
       title='Cost function over epochs')
ax.grid()


# In[874]:


# y = h(x)


# In[875]:


y_history = []
for i in range(m):
    y_history.append(h(G[i]))


# In[876]:


fig, ax = plt.subplots()
t= np.arange(0,m,1)
ax.plot(t,y_history)
ax.set(xlabel='X', ylabel='Cost function',
       title='Weight prediction over Perch Fish Lengths')
ax.grid()


# In[880]:


## Normal equation

theta = inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)

## Doesn't work either


# In[881]:


J()


# In[882]:


h(G[55])-y[20]

