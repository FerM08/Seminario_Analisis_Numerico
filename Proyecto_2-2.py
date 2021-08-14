#!/usr/bin/env python
# coding: utf-8

# In[11]:


#Importamos los paquetes que sean necesarios

import pandas as pd
import numpy as np
from scipy import integrate
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error, median_absolute_error
from scipy.integrate import odeint
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt
from PDE_params import PDEmodel 


# In[5]:


#Importamos los datos

data = pd.read_csv('CoV2019.csv')

#Clasificamos los datos que nos sirven

china= data["China"]
days= data["Days"]
deaths= data["Death China"]


# In[10]:


#Graficamos

plt.figure(figsize=(12, 8))
plt.title("Devolopment of COVID-19 in China  01/20/20-03/01/20")
plt.plot(days, china, '-*', label="Cases in China", color='pink')
plt.plot(days, deaths, '-*', label="Deaths in China", color='lightblue')
plt.yscale('log')
plt.ylabel("Cases")
plt.xlabel("Days after first WHO report")
plt.legend()


# In[13]:


#Condiciones iniciales

Hubei = 5917*10**4
Guangdong = 11346*10**4
Henan = 9605*10**4
Zhejiang = 5737*10**4
Hunan = 6899*10**4
Anhui = 6324*10**4
Jiangxi = 4648*10**4
N = 56*10**3                
init_I = 1
init_R = 1

conditions_array= [Hubei, Guangdong, Henan, Zhejiang, Hunan, Anhui, Jiangxi]

print(conditions_array)


# In[34]:


#Model 

data_1= pd.DataFrame(days, np.array[china, deaths])
print(data_1)


# In[ ]:


#Usamos la libreria PD_params 

__init__()

