#!/usr/bin/env python
# coding: utf-8

# # AUTHOR: ARYAMAN DAS

# ## GRIP-THE SPARKS FOUNDATION

# ### TASK-1: Prediction using Suprevised ML
#        The task is to predict the percentage of an student based on the no. of study hours.

# #### Importing The Modules

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression


# ## 1. Reading The Data.

# In[8]:


url = "http://bit.ly/w-data"
dataset = pd.read_csv(url)
print("Data imported.")


# In[10]:


dataset.head(10)


# In[11]:


dataset.describe()


# ## 2. Data Visualization.

# In[17]:


dataset.plot(x='Hours', y='Scores', style='.')
plt.title('Prediction Of Marks')
plt.xlabel('Hours of study.')
plt.ylabel('Marks obtained')
plt.show()


# ## 3. Training the Model.

# In[18]:


X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values


# In[23]:


X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.2, random_state=0)
Model1=LinearRegression()
Model1.fit(X_train.reshape(-1,1), Y_train)
print("Model is trained")


# ## 4. Plotting The Prediction Line (Line of Regression).

# In[24]:


print(Model1.intercept_)
print(Model1.coef_)


# In[25]:


line = Model1.coef_*X+Model1.intercept_

plt.scatter(X,Y)
plt.plot(X,line,color="red");
plt.show()


# ## 5. Predictions And Comparisons

# In[26]:


print(X_test) 
Y_pred = Model1.predict(X_test)


# In[27]:


# Comparison of Orignal vs Predicted Values
df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})  
df


# In[28]:


print('Training score:', Model1.score(X_train, Y_train))
print('Test score:', Model1.score(X_test, Y_test))


# In[29]:


df.plot(kind='bar',figsize=(5,5))
plt.grid(which='major', linewidth='0.5', color='red')
plt.grid(which='minor', linewidth='0.5', color='yellow')
plt.show()


# In[31]:


hours=9.25
test=np.array([hours])
test=test.reshape(-1,1)
own_pred=Model1.predict(test)
print("Number of Hours={}".format(hours))
print("Predicted Marks={}".format(own_pred[0]))


# ## 6. Final Analysis.

# In[32]:


from sklearn import metrics  
print('Mean Absolute Error:',metrics.mean_absolute_error(Y_test, Y_pred)) 
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
print('R-2:', metrics.r2_score(Y_test, Y_pred))


# In[ ]:




