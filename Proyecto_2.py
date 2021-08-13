#!/usr/bin/env python
# coding: utf-8

# In[14]:


#Importamos los paquetes necesarios

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[6]:


#Importamos los datos

datos = pd.read_csv('problem1.csv')

#Clasificamos datos

x_training = datos.iloc[:,0].values
y_training = datos.iloc[:, 1].values
x_test= datos.iloc[0:21, 2].values
y_test= datos.iloc[0:21, 3].values


# In[7]:


from sklearn.model_selection import train_test_split 

x_trained, x_tested, y_trained, y_tested = train_test_split(x_training, y_training, test_size=0.2, random_state=0)


# In[9]:


print(x_trained, x_tested, y_trained, y_tested)


# In[28]:


#Reshapeamos a x_training y hacemos regresión con polinomio de grado 4 

x_training_a= np.reshape(x_training, (-1,1))

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x_training_a)
pol_reg = LinearRegression()
pol_reg.fit(x_poly, y_training)

# Visualizing the Polymonial Regression results
def poly ():
    plt.scatter(x_training_a, y_training, color='pink')
    plt.plot(x_training_a, pol_reg.predict(poly_reg.fit_transform(x_training_a)), color='black')
    plt.title('Regresion')
    plt.show()
    return
poly()


# In[34]:


#Repetimos con grado 8

poly_reg_2 = PolynomialFeatures(degree=8)
x_poly_2 = poly_reg_2.fit_transform(x_training_a)
pol_reg_2 = LinearRegression()
pol_reg_2.fit(x_poly_2, y_training)

# Visualizing the Polymonial Regression results
def poly_2 ():
    plt.scatter(x_training_a, y_training, color='palegreen')
    plt.plot(x_training_a, pol_reg_2.predict(poly_reg_2.fit_transform(x_training_a)), color='black')
    plt.title('Regresion')
    plt.show()
    return
poly_2()


# In[64]:





# In[65]:


# Hacemos una prediccion con ambos polinomios
y_test_array= []
y_4= pol_reg.predict(poly_reg.fit_transform([y_test[:]]))


# In[46]:


y_8= pol_reg_2.predict(poly_reg_2.fit_transform([[2.2]]))


# In[22]:


# calculate aic for regression
mse = mean_squared_error(y_test[4], y_4)

def calculate_aic(mse):
	aic = 135 * log(mse) + 2
	return aic


# In[15]:


#Importamos los datos del problema 2

problema_2 = pd.read_csv('problem_2.csv')

#Clasificamos los datos

age= problema_2.x_age.values
chol= problema_2.x_cholesterol.values
sugar= problema_2.x_sugar.values
tcell= problema_2.x_Tcell.values
pro= problema_2.y.values


# In[21]:


#Correlacion entre variables

corr = problema_2.corr()
plt.figure(figsize=(12, 12))
sns.heatmap(corr, vmax=.8, square=True, linewidths=1, cmap="Set3")


# In[ ]:




