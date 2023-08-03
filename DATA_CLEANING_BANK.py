#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8) 
#Ajustamos los parametros de las figuras que ccrearemos
#matplotlib inlinehará que nuestros graficos aparezcan y se guarden en el notebook


# In[27]:


#Leemos la data
data = pd.read_csv(r'C:/Users/rodri/OneDrive/Documentos/PYTHON/DATASETS/datbank.csv')


# In[28]:


print(data.shape)
data.head()


# In[29]:


data.info()


# In[30]:


#Limpieza
#CASOS A TOMAR EN CUENTA A LA HORA DE LIMPIAR
#1: Datos faltantes en algunas celdas
#2: Columnas irrelevantes que no responden al problema del negocio que queremos resolver
#3: Registros o filas repetidas
#4: Valores extremos, pero ojo: cada caso debe ser atendido de manera específica
#5: Errores tipográficos en variables de texto (categóricas)


# In[31]:


#Al comparar la cantidad de datos por categoría frente al total nos damos cuenta de que hay pocos datos faltantes
#Para esto usamos la función .dropna(inplace=True) de pandas
data.dropna(inplace=True)
data.info()


# In[32]:


#Tipos de columnas irrelevantes
#1: Una columna que contiene información irrelevante para el objetivo del negocio (como el hobbie o deporte de un cliente)
#2: Una columna de texto o categórica de un solo valor (si todos son hombres, por ejemplo)
#3: Una columna numérica con un solo valor, ya sea 0, 1 o cualquiera
#4: Una columna con información redundante, como si hubiera la columna day y otra de montho, y otra mas llamada day-month
#Si tienes duda sobre la relevancia de la columna es mejor preguntar


# In[33]:


#Contaremos la cantidad de valores únicos o niveles que tiene cada columna categórica o de texto

cols_cat= ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']
for col in cols_cat:
    print(f'columna {col}: {data[col].nunique()} subniveles' )

#Ojo con la función data[col].nunique() que es la que nos devuelve la cantidad de datos unicos
#Lo que se hace aquí es basicamente imprimir o escribir con '' y poner una serie de resultados en medio, por eso usamos {}


# In[34]:


#Como podemos ver, cada columna tiene más de un nivel, así que no eliminaremos ninguna
#Procederemos con las columnas numericas


# In[35]:


data.describe()


# In[ ]:


#Solo al revisar la desviación estandar podemos ver que no tenemos columnas numéricas que contengan un solo valor, dado 
#que las desviaciones estandar son diferentes de 0
#Ahora borremos lo valore repetidos


# In[36]:


print(f'Data antes de eliminar duplicados: {data.shape}')
data.drop_duplicates(inplace=True)
print(f'Data después de eliminar duplicados: {data.shape}')


# In[37]:


#Ahora vamos a generar gráficas para las variables numericas para poder identificar los valores extremos o outliers

cols_num = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

fig, ax = plt.subplots(nrows=7, ncols=1, figsize=(8,30))
fig.subplots_adjust(hspace=0.5)

for i, col in enumerate(cols_num):
    sns.boxplot(x=col, data=data, ax=ax[i])
    ax[i].set_title(col)


# In[ ]:


#Como podemos ver, en el campo de edad hay edades que sobrepasan los 100 años, lo cual no es normal
#Por eso se eliminarán los valores extremos en ese caso
#En balance todo está bien, el valor de 5000 es normal cuando hablamos de dinero
#En la columna day también, están entre 0 y 31
#En la columna duration tenemos problemas porque no puede haber duración negativa, entonces esos valores se elimnarán
#En campaign, pdays y previous todo bien


# In[38]:


print(f'Data antes de eliminar edades mayores a 100: {data.shape}')
data = data[data['age']<=100]
print(f'Data después de eliminar edades mayores a 100: {data.shape}')


# In[39]:


print(f'Data antes de eliminar duraciones de llamada negativas: {data.shape}')
data = data[data['duration']>0]
print(f'Data después de eliminar duraciones de llamada negativas: {data.shape}')


# In[40]:


print(f'Data antes de eliminar registros de previous: {data.shape}')
data = data[data['previous']<=100]
print(f'Data después de eliminar registros de previous: {data.shape}')


# In[41]:


#Ahora vamos a generar gráficas para las variables categóricas para ver si hay niveles demás

cols_cat = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']

fig, ax = plt.subplots(nrows=10, ncols=1, figsize=(10,30))
fig.subplots_adjust(hspace=1)

for i, col in enumerate(cols_cat):
    sns.countplot(x=col, data=data, ax=ax[i])
    ax[i].set_title(col)
    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=30)


# In[42]:


#Como podemos observar, hay categorías como job, en la que tenemos varios niveles que significan lo mismo, pero están escritas de manera diferente
#Así pasa en varias categorías. Para ello, lo primero que haremos es homogeneizarlas lo más que podamos
#Primero haremos que todas estén escritas en minúsculas

for column in data.columns:
    if column in cols_cat:
        data[column] = data[column].str.lower()

fig, ax = plt.subplots(nrows=10, ncols=1, figsize=(10,30))
fig.subplots_adjust(hspace=1)

for i, col in enumerate(cols_cat):
    sns.countplot(x=col, data=data, ax=ax[i])
    ax[i].set_title(col)
    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=30)


# In[ ]:


#Como podemos ver, ahora en la columna Job, por ejemplo, ya no está el nivel managment en mayuscula
#En marital tampoco está DIVORCED
#Sin embargo, aun hay niveles que están escritos de formas diferentes, aquellos, los tendremos que reemplazar uno por uno


# In[43]:


#Reemplazaremos admin. por administrative en la columna job
print(data['job'].unique())
data['job'] = data['job'].str.replace('admin.', 'administrative', regex=False)
print(data['job'].unique())


# In[44]:


#Reemplazaremos div. por divorced en la columna marital
print(data['marital'].unique())
data['marital'] = data['marital'].str.replace('div.', 'divorced', regex=False)
print(data['marital'].unique())


# In[45]:


#Reemplazaremos phone por telephone en la columna contact
print(data['contact'].unique())
data[data['contact'] == 'phone'] = 'telephone'
print(data['contact'].unique())


# In[46]:


#Reemplazaremos unk por unknown en la columna poutcome
print(data['poutcome'].unique())
data[data['poutcome'] == 'unk'] = 'unknown'
print(data['poutcome'].unique())


# In[47]:


#Ahora veremos el tamaño de la base de datos final, luego de realizada la limpieza
data.shape


# In[48]:


#finalmente lo guardamos en su carpeta original

ruta = 'C:/Users/rodri/OneDrive/Documentos/PYTHON/DATASETS/datbank.csv'
data.to_csv(ruta)


# In[ ]:




