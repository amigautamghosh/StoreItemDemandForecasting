
# coding: utf-8

# In[18]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd

# Read train.csv file and set datatype
data_type = {'store': 'int8', 'item': 'int8', 'sales': 'int16'}
df = pd.read_csv("C:/Users/gautam.ghosh/Desktop/ML-Data/train.csv/train.csv", parse_dates= ['date'], dtype= data_type)
df.head()


# In[7]:


test = pd.read_csv('C:/Users/gautam.ghosh/Desktop/ML-Data/test.csv/test.csv', parse_dates=['date'], index_col='id')
test.head()


# In[8]:


print("Train shape:", train.shape)
print("Test shape:", test.shape)


# In[4]:


df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day']=df['date'].dt.day
###df['week'] = df['date'].dt.week

##print(df)
df['weekofyear'] = df['date'].dt.weekofyear
df['dayofweek'] = df['date'].dt.dayofweek
##df['day']=df['date'].dt.day
##df['dayofweek'] = df['date'].dt.dayofweek
##df['weekday'] = df['date'].dt.weekday
df['weekday_name'] = df['date'].dt.weekday_name
df['dayofyear']=df['date'].dt.dayofyear
df['quarter'] = df['date'].dt.quarter
df.drop('date', axis=1, inplace=True)
print(df)


# In[5]:


print(df)


# In[19]:


df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day']=df['date'].dt.day
df['weekofyear'] = df['date'].dt.weekofyear
df['dayofweek'] = df['date'].dt.dayofweek
df['weekday_name'] = df['date'].dt.weekday_name
df['dayofyear']=df['date'].dt.dayofyear
df['quarter'] = df['date'].dt.quarter
df['weekend'] = ((df['dayofweek']) // 5 == 1).astype(float)

df.head()


# In[20]:


df = pd.get_dummies(df)
df.head()


# In[21]:


df["mean-store-item"] = df.groupby(["item", "store"])["sales"].transform("mean")
df["median-store-item"] = df.groupby(["item", "store"])["sales"].transform("median")

df["mean-month-item"] = df.groupby(["month", "item"])["sales"].transform("mean")
df["median-month-item"] = df.groupby(["month", "item"])["sales"].transform("median")
df["sum-month-item"] = df.groupby(['month',"item"])["sales"].transform("sum")

df["median-month-store"] = df.groupby(["month", "store"])["sales"].transform("median")
df["mean-month-store"] = df.groupby(["month", "store"])["sales"].transform("mean")
df["sum-month-store"] = df.groupby(['month',"store"])["sales"].transform("sum")

df["mean-item"] = df.groupby(["item"])["sales"].transform("mean")
df["median-item"] = df.groupby(["item"])["sales"].transform("median")

df["mean-store"] = df.groupby(["store"])["sales"].transform("mean")
df["median-store"] = df.groupby(["store"])["sales"].transform("median")

df["mean-month-item-store"] = df.groupby(['month',"item","store"])["sales"].transform("mean")
df["mean-month-item-store"] = df.groupby(['month',"item","store"])["sales"].transform("median")

df["mean-weekofyear-item-store"] = df.groupby(['weekofyear',"item","store"])["sales"].transform("mean")
df["median-weekofyear-item-store"] = df.groupby(['weekofyear',"item","store"])["sales"].transform("median")
df["sum-weekofyear-item-store"] = df.groupby(['weekofyear',"item","store"])["sales"].transform("sum")
    
df["mean-quarter-store"] = df.groupby(['quarter',"store"])["sales"].transform("mean") 
df["median-quarter-store"] = df.groupby(["quarter", "store"])["sales"].transform("median")
df["sum-item-store"] = df.groupby(['quarter',"store"])["sales"].transform("sum")

df["mean-quarter-item"] = df.groupby(["quarter", "item"])["sales"].transform("mean")
df["median-quarter-item"] = df.groupby(["quarter", "item"])["sales"].transform("median")
df["sum-item-quarter"] = df.groupby(['quarter',"item"])["sales"].transform("sum") 

df["mean-weekend-item-store"] = df.groupby(['weekend',"item","store"])["sales"].transform("mean")
df["median-weekend-item-store"] = df.groupby(['weekend',"item","store"])["sales"].transform("median")

df.head()


# In[22]:


col = [i for i in df.columns if i not in ['sales', 'date']]
print(col)


# In[23]:


y = 'sales'


# In[24]:


train = df.loc[~df.sales.isna()]
train.head()


# In[25]:


df.drop('date', axis=1, inplace=True)
df.head()


# In[26]:


from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(train[col],train[y], test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
X_train.head()


# In[27]:


from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)


# In[28]:


from sklearn.metrics import mean_absolute_error
predictions = reg.predict(X_test)
print(mean_absolute_error(predictions,y_test))


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor(n_estimators=100, n_jobs=-1)
RF.fit(X_train, y_train)


# In[ ]:


from sklearn.metrics import mean_absolute_error
predictions = RF.predict(X_test)
print(mean_absolute_error(predictions,y_test))


# In[ ]:


print(predictions)

