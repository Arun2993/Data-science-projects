#!/usr/bin/env python
# coding: utf-8

# In[1]:


## ML Algorithm, Deep Learning, NLP - Chatbots, GenAI & AgenticAI*****
### Data Pipeline - Data collection , cleaning, Analysis - EDA, Modeling, Deployment(Dev, QA,Prod), Monitoring
#### Applications : Delay Prediction, Dynamic Pricing(fare pricing), Maintenance Prediction, Sentiment Analysis,
  #### Chatbots and virtual assistants, Passenger Personalization - 

### Flight Delay Prediction using ML


# # Import basic library

# In[2]:


import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[3]:


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 25)


# # importing dataset

# In[4]:


dataset = pd.read_excel("airline_dataset.xlsx")
dataset.head(25)


# In[5]:


dataset.info()


# # EDA - dtale

# In[8]:


pip install dtale


# In[9]:


import dtale


# In[10]:


dtale.show(dataset)


# In[11]:


dataset.isnull().sum()/len(dataset)*100


# In[12]:


dataset.dropna(inplace=True)


# In[13]:


dataset.columns


# In[14]:


dataset['Airline'].unique()




dataset['Airline'].value_counts()



dataset.info()



dataset['jouney_day'] = pd.to_datetime(dataset.Date_of_Journey, format="%d/%m/%Y").dt.day



dataset['jouney_month'] = pd.to_datetime(dataset.Date_of_Journey, format="%d/%m/%Y").dt.month



dataset['jouney_year'] = pd.to_datetime(dataset.Date_of_Journey, format="%d/%m/%Y").dt.year



dataset.Date_of_Journey



dataset.head()



dataset['jouney_year'].value_counts()



dataset.drop(['Date_of_Journey','jouney_year'], axis=1, inplace=True)



dataset.head()



dataset.info()



dataset['dep_hour'] = pd.to_datetime(dataset['Dep_Time']).dt.hour
dataset['dep_min'] = pd.to_datetime(dataset['Dep_Time']).dt.minute



dataset.head()



dataset.drop(['Dep_Time'], axis=1, inplace=True)



dataset.head()



dataset['arr_hour'] = pd.to_datetime(dataset['Arrival_Time']).dt.hour
dataset['arr_min'] = pd.to_datetime(dataset['Arrival_Time']).dt.minute



dataset.drop(['Arrival_Time'], axis=1, inplace=True)



dataset.head(2)



dataset.info()



list(dataset['Duration'])



len(list(dataset['Duration']))



duration = list(dataset['Duration'])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"
        else:
            duration[i] = "0h " + duration[i]

duration_hours = []
duration_mins = []

for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))



dataset['duration_hour'] = duration_hours
dataset['duration_mins'] = duration_mins



dataset.head()



dataset.drop(['Duration'], axis=1, inplace=True)



dataset.head(2)



dataset.info()



dataset['Airline'].value_counts()



sns.catplot(y='Price', x='Airline', data=dataset.sort_values('Price', ascending=False), kind='boxen', height=8, aspect=4)
plt.show()



Airline1 = dataset[['Airline']]
Airline1 = pd.get_dummies(Airline1, drop_first=True, dtype='int64')
Airline1.head()



dataset['Source'].value_counts()



sns.catplot(y='Price', x='Source', data=dataset.sort_values('Price', ascending=False), kind='boxen', height=6, aspect=3)
plt.show()



Source1 = dataset[['Source']]
Source1 = pd.get_dummies(Source1, drop_first=True, dtype='int64')
Source1.head()


dataset.head(2)

dataset['Destination'].value_counts()



sns.catplot(y='Price', x='Destination', data=dataset.sort_values('Price', ascending=False), kind='boxen', height=6, aspect=3)
plt.show()

dataset["Destination"] = np.where(dataset["Destination"]=='New Delhi','Delhi',dataset["Destination"])

Destination1 = dataset[['Destination']]
Destination1 = pd.get_dummies(Destination1, drop_first=True, dtype='int64')
Destination1.head()

dataset.head()


dataset['Route'].value_counts()


dataset['Route'].nunique()

dataset.drop(['Route'], axis=1, inplace=True)


dataset['Total_Stops'].value_counts()

dataset.replace({"non-stop": 0,"1 stop":1, "2 stops":2,"3 stops":3,"4 stops":4}, inplace=True)


dataset['Total_Stops'].value_counts()


dataset.head(2)


dataset['Additional_Info'].value_counts(normalize=True)


dataset.drop(['Additional_Info'], axis=1, inplace=True)

dataset.head(2)


final_dataset = pd.concat([dataset, Airline1,Source1, Destination1], axis=1)


final_dataset.head(2)

final_dataset.drop(['Airline','Source','Destination'], axis=1, inplace=True)


final_dataset.head()


# split the data into dep and indep variables
x = final_dataset.drop(['Price'],axis=1)
y = final_dataset['Price']


x.head()

y.head()


plt.figure(figsize=(20,16))
sns.heatmap(final_dataset.corr(), annot=True, cmap="RdYlGn")
plt.show()

# To check which variable is more significant to impact price - Feature Importance.


from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(x,y)

selection.feature_importances_


plt.figure(figsize=(15,8))
feature_imp = pd.Series(selection.feature_importances_, index=x.columns)
feature_imp.nlargest(20).plot(kind='barh')
plt.show()


# # Building Model - Random Forest Model


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1)

from sklearn.ensemble import RandomForestRegressor
ref_reg = RandomForestRegressor()
ref_reg.fit(x_train, y_train)


y_pred_train = ref_reg.predict(x_train)
y_pred_test = ref_reg.predict(x_test)


ref_reg.score(x_train, y_train)


ref_reg.score(x_test, y_test)


# In[81]:


from sklearn import metrics


# In[82]:


print('MAE :', metrics.mean_absolute_error(y_test, y_pred_test))
print('MSE :', metrics.mean_squared_error(y_test, y_pred_test))
print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_pred_test)))


# In[83]:


metrics.r2_score(y_test, y_pred_test)


# In[84]:


# RMSE / (max(y) -min(y))
2022.61/(max(y) - min(y))


# # HyperParameter Tunning
# ### RandomizedSearchCV and GridSearchCV

# In[ ]:





# In[85]:


# Save the model to reuse it again while deploy the model
import pickle
file = open('price_prediction.pkl', 'wb')
pickle.dump(ref_reg, file)


# In[86]:


model = open('price_prediction.pkl','rb')
rforest = pickle.load(model)


# In[87]:


x_test.head()


# In[88]:


y_prediction = rforest.predict(x_test)


# In[89]:


y_prediction



metrics.r2_score(y_test, y_prediction)




