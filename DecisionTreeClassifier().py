#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data = pd.read_csv('kyphosis.csv')
data.head()


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


x = data.drop('Kyphosis', axis=1)
x.head()


# In[7]:


y = data['Kyphosis']
y.head()


# In[9]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)


# In[11]:


x_train.shape


# In[14]:


x_test.shape


# # Decision Tree

# In[15]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train, y_train)


# In[23]:


pred = model.predict(x_test)


# In[21]:


y_test


# In[24]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, pred)
accuracy


# In[25]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)
cm


# In[ ]:




