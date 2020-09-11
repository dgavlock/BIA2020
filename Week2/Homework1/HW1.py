#!/usr/bin/env python
# coding: utf-8

# In[120]:


import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[121]:


file = scipy.io.loadmat('HW1_Q1.mat')


# In[122]:


data = file['f']
x = np.linspace(1, len(data), 100)


# In[154]:


ones = np.ones(100)


# In[123]:


fit = np.polyfit(x, data, deg=1)


# In[124]:


fit


# In[125]:


alpha, beta = fit[0][0], fit[1][0]


# In[158]:


y_hat = [alpha*i + beta for i in x]


# In[159]:


plt.plot(x, data, x, y_hat)
plt.title('Linear Function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['data', 'fit'])
plt.savefig(fname='LinearFunc.png')


# ## For Quadratic function

# In[113]:


fit2 = np.polyfit(x, data, deg=2)


# In[114]:


fit2[2][0]


# In[115]:


alpha1, beta1, gamma1 = fit2[0][0], fit2[1][0], fit2[2][0]


# In[116]:


y_hat_2 = [gamma1*i**2 + alpha1*i + beta1 for i in x]


# In[141]:


plt.plot(x, data, x, y_hat_2)
plt.title('Quadratic Function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['data', 'fit'])
plt.savefig('QuadraticFunc.png',format = 'png', dpi=300)

