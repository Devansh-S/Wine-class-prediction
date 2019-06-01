
# coding: utf-8

# In[57]:


#loading libraries and dependencies

from sklearn.datasets import load_wine
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[58]:


#loading the dataframe

df = load_wine()


# In[59]:


#splitting the data into parts 

data = df.data
target = df.target
desc = df.DESCR


# In[60]:


#printing the first row of data and its class 
#printing the dataset description

print(data[0])
print(target[0])
print(desc)


# In[123]:


#splitting the dataset into training and testing datasets
#printing the shapes of datasets so formed

x_train,x_test,y_train,y_test = train_test_split(data, target , test_size = 0.15)
print (x_train.shape)
print (y_train.shape) 
print (x_test.shape)
print (y_test.shape)


# In[124]:


#using ExtraTreesClassifier model
model = ExtraTreesClassifier(bootstrap=False,min_impurity_decrease=0.01)

#fitting the training data into hte model
model.fit(x_train,y_train)

#predicting the values using the model
y_pred = model.predict(x_test)

#calculating and printing the accuracy metrics.
score = accuracy_score(y_pred,y_test)
print("ACCURACY = " + str((score)*100))

