#!/usr/bin/env python
# coding: utf-8

# In[64]:


#Importamos los paquetes necesarios

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb


# In[4]:


#Importamos los datos

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample= pd.read_csv('sample_submission.csv')

ids = test['Id'].values


# In[6]:


#Vemos los datos 

train.describe()


# In[7]:


test.describe()


# In[8]:


sample.describe()


# In[14]:


#Distribucion de Sales Price

sns.distplot(train['SalePrice'],color="pink")


# In[17]:


#Separar Sale price y hacerla una variable de Python sola

sale_Price   = train.SalePrice.values
train_Sin_SP = train.drop('SalePrice', 1)


# In[22]:


#Correlacion entre variables

corr = train.corr()
plt.figure(figsize=(12, 12))
sns.heatmap(corr, vmax=.8, square=True, linewidths=.5, cmap="Set3")


# In[30]:


indexes = train[(train['GrLivArea']>4000) & (train['SalePrice']<200000)].index 

x_train = train.drop(indexes)
y_train = np.delete(sale_Price, indexes)


# In[38]:


#Declaramos el modelo 

model_xgb = xgb.XGBRegressor(n_estimators=2200)


# In[39]:


#Los folds y la metrica con la evaluacion

n_folds = 5

def rmsle(model_xgb):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

x_train.shape 


# In[40]:


#Refinamiento

model_xgb.fit(x_train, y_train)
xgb_train_pred = model_xgb.predict(x_train)
xgb_pred = model_xgb.predict(df_test)
print(rmsle(y_train, xgb_train_pred))


# In[43]:


#Otro index

indexes_2 = train[(train['OverallQual']>4000) & (train['SalePrice']<200000)].index 

x_train_2 = train.drop(indexes)
y_train_2 = np.delete(sale_Price, indexes)


# In[58]:


#Quitamos los valores no float64 de los 2 index x


#x_train
non_Floats_x_train=[]

for col in x_train:
    if x_train[col].dtypes != "float64":
        non_Floats_x_train.append(col)
x_train = x_train.drop(columns=non_Floats_x_train)


#x_train_2
non_Floats_x_train_2=[]

for col in x_train_2:
    if x_train_2[col].dtypes != "float64":
        non_Floats_x_train_2.append(col)
x_train_2 = x_train_2.drop(columns=non_Floats_x_train_2)


# In[70]:


#Modelo_2

#Declaramos el modelo 

model_xgb_2 = xgb.XGBRegressor(verbosity = 0)

model.fit(x_train, y_train)

#Evaluamos

cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1)

scores = cross_val_score(model_xgb_2, x_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)


print('Scores: %.3f (%.3f)' % (scores.mean(), scores.std()) )


# In[71]:


#Modelo_2.1

#Declaramos el modelo 

model_xgb_2_1 = xgb.XGBRegressor(verbosity = 0)

model.fit(x_train_2, y_train_2)

#Evaluamos

cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1)

scores = cross_val_score(model_xgb_2_1, x_train_2, y_train_2, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)


print('Scores: %.3f (%.3f)' % (scores.mean(), scores.std()) )


# In[87]:


indexes_2 = test[(test['GrLivArea']>4000) & (train['SalePrice']<200000)].index 

x_test = test.drop(indexes)
y_test = np.delete(sale_Price, indexes)
x_test_a= np.asarray(x_test)


# In[85]:


#Predict

preds = model_xgb_2.predict(x_test_a)


# In[ ]:




