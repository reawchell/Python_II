#!/usr/bin/env python
# coding: utf-8

# ## Proyecto Machine Learning

# Proyecto Final: Propuesta Machine Learning
# 
# El siguiente proyecto tiene como finalidad aplicar diferentes algoritmos de clasificación y regresión de aprendizaje
# supervisado, por ejemplo: KNN, Naive Bayes, Linear Regression, Logistic Regression, Support Vector Machine (SVMs),
# Decission Trees, Random Forest, etc.
# Objetivos:
# - Se deben aplicar tres(3) métodos de clasificación: los datos se deben clasificar según el tipo de alojamiento,
# definido en el campo room_type, a partir del resto de características. Es decir, en room_type estarán
# codificadas las clases y en el resto de los campos los atributos.
# - La tarea de regresión consistirá en considerar el precio por noche como variable dependiente, y el resto de
# los campos como variables independientes. Es decir, se tratará de predecir los valores del precio del
# alojamiento, a partir de los atributos.
# Los datos:
# - Los datos provienen de la web InsideAirBnB, dedicada al estudio de los alquileres vacacionales
# ofrecidos en la plataforma AirBnB. El fichero AB_NYC_2019.csv. 
# 

# ### Importacion de Librerias

# In[2]:


import os
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import pylab
import pingouin as pg
import urllib
import folium
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn import preprocessing 
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import lime
import lime.lime_tabular
from sklearn.metrics import mean_squared_error, r2_score
import eli5
from scipy.stats import boxcox
from scipy.special import boxcox1p
from scipy.special import inv_boxcox
from sklearn.preprocessing import PowerTransformer, RobustScaler
from collections import Counter
from sklearn.preprocessing import PowerTransformer
from folium.plugins import HeatMap
from folium.plugins import FastMarkerCluster
from scipy import stats
import seaborn as sns
import plotly.express as px
import matplotlib.colors as c
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from scipy.special import boxcox1p
from scipy.special import inv_boxcox
from scipy import stats
import random
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.colors import n_colors
from plotly.subplots import make_subplots
from mpl_toolkits.mplot3d import axes3d
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from matplotlib import cm
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore")


# In[3]:


PATH_FILE = os.getcwd()
PATH_DATA = r"C:\Users\raque\Desktop\PYTHON\GLOSARIO CLASES\data"
os.chdir(PATH_DATA)


# In[4]:


os.getcwd()


# ### Lectura de Fichero

# In[5]:


df = pd.read_csv("AB_NYC_2019.csv")
df.head(10)


# In[6]:


df


# ### Información del dataset

# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df.columns


# In[10]:


print('Características cuantitativas: {}'.format(len([d for d in df.columns if df.dtypes[d] != 'object'])))
print('Características cualitativas: {}'.format(len([d for d in df.columns if df.dtypes[d] == 'object'])))


# ### Datos estadísticos de Price

# In[11]:


df['price'].describe()


# In[12]:


fig = plt.figure(figsize=(16,6))

ax1 = fig.add_subplot(121)
sns.boxplot(y = df['price'], ax=ax1, color='yellow')
describe = df['price'].describe().to_frame().round(2)

ax2 = fig.add_subplot(122)
ax2.axis('off')
font_size = 16
bbox = [0, 0, 1, 1]
table = ax2.table(cellText = describe.values, rowLabels = describe.index, bbox=bbox, colLabels=describe.columns)
table.set_fontsize(font_size)
fig.suptitle('Distribución de Precios con visual de outliers)', fontsize=16)
plt.show()


# ### Cantidad de alojamientos por zonas

# In[13]:


listings = df['neighbourhood_group'].value_counts()
print(listings)


# ### La cantidad de alojamientos que existe para cada tipo de alojamiento (room_type) cantidad y porcentaje 

# In[14]:


df.room_type


# ### Cantidad de tipos de alojamientos

# In[15]:


df['room_type'].value_counts()


# ### El gráfico % alojamientos que existe para cada tipo de alojamiento (room_type) 

# In[16]:


plt.figure(figsize=(12,7))
sns.countplot(y='room_type', data=df)
plt.title("Tipos de Alojamientos", size=24)
plt.show()


# In[17]:


#pd.options.display.float_format = "{:.2f}".format


# En la distribución de precios podemos observar que el máximo se situa en 10.000, el mínimo en 0 mientras que la media ronda los 152$, datos que pueden indicar error en algunas entradas de precios.

# In[18]:


df['price'].describe()


# ### El gráfico de dispersión del atributo precio.

# In[19]:


sns.catplot("price",data=df,  kind="strip", aspect=2)
plt.title("Precios", size=24) 
plt.show()


# ### Gráfico de caja (boxplot) para precio con visual de datos atípicos

# In[20]:


plt.figure(figsize=(15,10))
plt.boxplot(df["price"], vert=False)
plt.title("Precios", size=24)
plt.show()


# ### Datos numéricos del dataset 

# In[21]:


sns.set_style('whitegrid')
df.plot.hist(bins=100, figsize=(15,8)) 
plt.title("Datos Numéricos", size=24)
plt.show()


# ### Matriz de correlación de los datos numéricos

# In[22]:


sns.pairplot(df)                 
plt.show()  


# ### Matriz de correlación Mapa de Calor 

# In[22]:


corr = df.corr(method='pearson')
plt.figure(figsize=(15,12)) 
plt.xticks(range(df.shape[1]), df.columns, fontsize=13, rotation=45)
plt.yticks(range(df.shape[1]), df.columns, fontsize=13)
sns.heatmap(corr, annot=True, fmt=".2f", vmax=.3, center=0, 
            square=True, linewidths=.5, cbar_kws={"shrink": .5}).set(ylim=(10, 0))
df.columns
plt.title("Matriz de Correlación - Mapa de calor", size=24)
plt.show()


# In[23]:


df_corr = df.corr(method='pearson') 
df_corr 


# Revisión de los datos nulos en nuestra data, mediante la suma vemos que name y host-name tienen pocos nulos, mientras que  last_review y review_per_month tienen un total de 10052 cada uno. Mas adelante trataremos de encontrar una solucion

# In[24]:


df.isnull().sum()


# En el siguiente gráfico podemos ver que los precios mas elevados y los aparentemente atípicos se encuentran en alojamientos del tipo Private_room y Entire home/apt. La gran mayoria de Private_room está en un rango de 2000$, Entire Home/apt sore unos 4100$ y Shared_room no supera nunca los 2000$.

# In[25]:


plt.figure(figsize=(15,12))
sns.scatterplot(x='room_type', y='price', data=df)

plt.xlabel("Room Type", size=13)
plt.ylabel("Price", size=13)
plt.title("Precios por Tipos de Alojamientos",size=15, weight='bold')
plt.show()


# En este gráfico se observa que los precios mas altos se localizan en Brooklin y Manhathan, ya sea en Private_room como en Entire Home/apt, lo cual era de esperar por la zona y los tipos de alojamientos, por otra parte Manhathan tambien ofrece alojamientos compartidos com menor precio.

# In[26]:


plt.figure(figsize=(20,15))
sns.scatterplot(x="room_type", y="price",
            hue="neighbourhood_group", size="neighbourhood_group",
            sizes=(50, 200), palette="RdYlGn", data=df)

plt.xlabel("Room Type", size=13)
plt.ylabel("Price", size=13)
plt.title("Precios por Tipos de Alojamientos y Barrios",size=15, weight='bold')
plt.show()


# El gráfico a seguir muestra la cantidad de críticas por barrios y precios. Se puede ver que el grueso de las mismas están situadas en un rango de 2000$. Esto puede ser por que los que pagan menos son mas propensos a dejar un comentario ya sea positivo o negativo, mientras que los que pagan mas no, aun que lo mas probable es que nuestra data tenga mayoria de alojamientos que rodan esos valores.

# In[27]:


plt.figure(figsize=(20,15))
sns.set_palette("Paired")

sns.lineplot(x='price', y='number_of_reviews', 
             data=df[df['neighbourhood_group']=='Brooklyn'],
             label='Brooklyn')
sns.lineplot(x='price', y='number_of_reviews', 
             data=df[df['neighbourhood_group']=='Manhattan'],
             label='Manhattan')
sns.lineplot(x='price', y='number_of_reviews', 
             data=df[df['neighbourhood_group']=='Queens'],
             label='Queens')
sns.lineplot(x='price', y='number_of_reviews', 
             data=df[df['neighbourhood_group']=='Staten Island'],
             label='Staten Island')
sns.lineplot(x='price', y='number_of_reviews', 
             data=df[df['neighbourhood_group']=='Bronx'],
             label='Bronx')
plt.xlabel("Price", size=13)
plt.ylabel("Number of Reviews", size=13)
plt.title("Comentarios por Precios y Barrios",size=15, weight='bold')
plt.show()


# A seguir vemos que el Alojamiento con mayor número de criticas, un total de 629, está en Queens, és del tipo Private_room con un precio de 47$ la noche.

# In[28]:


crit_sup=df.nlargest(20,'number_of_reviews')
crit_sup


# ### Média de precios por noche 

# In[29]:


price_avrg=crit_sup.price.mean()
print('Precio x Noche: {}'.format(price_avrg))


# ### Anfitriones con más alojamientos disponibles

# In[30]:


top_host=df.host_id.value_counts().head(10)
top_host


# A continuación vamos a revisar la columa Name con la idea de verificar si el nombre puede tener algun tipo de importancia

# In[31]:


_names_=[]
for name in df.name:
    _names_.append(name)  
def split_name(name):
    spl=str(name).split()
    return spl

_names_for_count_=[]
for x in _names_:
    for word in split_name(x):
        word=word.lower()
        _names_for_count_.append(word)


# In[32]:


pal_pop=Counter(_names_for_count_).most_common()
pal_pop=pal_pop[0:15]


# In[33]:


pal=pd.DataFrame(pal_pop)
pal.rename(columns={0:'Words', 1:'Count'}, inplace=True)


# Ahora podemos ver las palabras mas comunes. No parece que puedan tener mucho peso a la hora de valorar el precio con lo cual no realizaremos mucho mas con esta columna.

# In[34]:


plt.figure(figsize=(15,12))
viz_5=sns.barplot(x='Words', y='Count', data=pal)
viz_5.set_title('Palabras mas Utilizadas')
viz_5.set_ylabel('Count of words')
viz_5.set_xlabel('Words')
viz_5.set_xticklabels(viz_5.get_xticklabels(), rotation=90)


# ### Media de Precios por Barrios

# In[35]:


df.groupby(["neighbourhood_group"])['price'].mean()


# Utilizando un mapa de calor se puede observar donde están localizados por latitud y longitud

# In[36]:


mapa = folium.Map([40.730610,-73.935242],zoom_start=10)
HeatMap(df[['latitude','longitude']],radius=10).add_to(mapa)
display(mapa)


# El siguiente mapa resulta muy interesante ya que muestra por cluster's la disponibilidad de alojamientos, da mucho juego a la hora de visualizar y confieso que pasé un rato con el

# In[37]:


cluster_map = folium.Map([40.730610,-73.935242],zoom_start=10)
FastMarkerCluster(df[['latitude','longitude']],radius=10).add_to(cluster_map)
cluster_map


# ###  Cantidad de Alojamientos por Barrios,

# Williamsburg retiene casi 4000 del total de alojamientos disponibles, seguido de Bedford-Stuyvesant con 3714

# In[39]:


listings = df['neighbourhood'].value_counts()
print(listings)


# Graficamos los 20 Barrios con más Alojamientos

# In[40]:


plt.figure(figsize=(24,10))
top_bar= sns.countplot(df['neighbourhood'],order=df['neighbourhood'].value_counts().index[0:20],palette='bright')
top_bar.set_xticklabels(top_bar.get_xticklabels(),rotation=90, size=18)
plt.title('20 Barrios con más Alojamientos',size=20,pad=10)
plt.show()


# Bueno, parece que ahora ya conocemos mejor nuestros datos.
# 
# Sabemos que tratamos con alojamientos disponibles na la ciudad de NY ofertados por Airbnb, estos estan divididos por tipos, Habitacion Privada, Apartamento entero y alojamiento compartido, donde las dos primeras son mas caras aunque mas populares.
# Sabemos que los barrios con precios mas altos son Manhathan y Brooklin mientras que el Bronx es mas barato.
# Existe mayor frecuencia de críticas en los alojamientos con precios no superiores a 2000$.
# Nuestra data muestra valores atípicos en precios siendo el mínimo 0 y el maximo 10.000$, tambien tenemos datos faltantes y para ambos detalles encontraremos una solución.

# Hora de empezar a organizar, limpiar y rellenar

# Empezamos por verificar los datos faltantes por columnas con un total y el porcentaje que representa.

# In[41]:


total = df.isnull().sum().sort_values(ascending = False)
porcentaje = (df.isnull().sum() / df.isnull().count()).sort_values(ascending = False)
missing_data = pd.concat([total, porcentaje], axis = 1, keys = ['Total', 'Porcentaje'])
missing_data.head(20)


# Muy bien, en el caso de reviews_per_month sustituiremos los datos faltantes por la media, ya que son pocos y no parece que pueda afectar al modelo

# In[42]:


mean = df['reviews_per_month'].mean()
df['reviews_per_month'].fillna(mean, inplace=True)
df.isnull().sum()


# El caso de name lo que haremos será directamente eliminar los 16 datos faltantes

# In[43]:


df = df.drop(df.loc[df['name'].isnull()].index)
df.isnull().sum()


# Optaremos por descartar id, host_name y last_reviwew. El motivo es que no aportan mucho valor para el cometido que es predecir el precio

# In[44]:


df = df.drop(columns=['id','host_name','last_review'])
df.isnull().sum()


# Ahora vamos a revisar la variable precios, primero veremos los alojamientos que contemplan un precio de 0$ lo que casi seguro trata de un error de entrada en los datos

# In[45]:


df[df['price']==0].head()


# In[46]:


df[df['price']==0].index


# Visto que son 11 los registros optamos por descartarlos

# In[47]:


df.drop(df[df['price']==0].index, inplace=True)


# Ahora vamos a ver los precios superiores a 8000$ por noche

# In[48]:


df[df['price']>=8000].head(100)


# In[49]:


df[df['price']>=8000].index


# En este caso no está muy claro que todos sean un error de entrada, ya que en algunos casos el minimo de noches supera las 30 ademas de estar localizados en barrios que ya vimos tienen pernotaciones caras ademas de tratarse de Private_room y Entire Home/apt. Lo que si podemos ver es que se trata de una muestra pequeña pero que puede afectar a nuestro objetivo, con lo cual tambien los descartamos.

# In[50]:


df.drop(df[df['price']>=8000].index, inplace=True)


# In[51]:


df.reset_index(drop=True, inplace=True)


# Ahora vamos a categorizar las variables neihbourhood_group, neighbourhood, room_type y name

# In[52]:


df['neighbourhood_group']= df['neighbourhood_group'].astype("category").cat.codes
df['neighbourhood'] = df['neighbourhood'].astype("category").cat.codes
df['room_type'] = df['room_type'].astype("category").cat.codes
df['name'] = df['name'].astype("category").cat.codes
df.info()


# In[53]:


sns.set_style('whitegrid')
df.plot.hist(bins=100, figsize=(15,8)) 
plt.title("Datos Numéricos", size=24)
plt.show()


# Nos toca revisar y tratar nuesta variable objetivo, Precio.
# En el siguiente gráfico vemos que tiene una asimetria positiva.

# In[54]:


plt.figure(figsize=(10,10))
sns.distplot(df['price'], fit=stats.norm)
plt.title("Distribución Precios",size=15, weight='bold')
plt.show()


# Para poder solventar el problema que tenemos de asimetria positiva y poder equiparar lo maximo posible a una normalidad gausiana utilizaremos el método Box-Cox

# Transformación Yeo Johnson

# In[55]:


pt = PowerTransformer(method='yeo-johnson')


# In[56]:


df['price'],price_lambda = boxcox(df['price'])
print('"price lambda": {}'.format(price_lambda))


# Y vualá

# In[57]:


plt.figure(figsize=(10,10))
sns.distplot(df['price'], fit=stats.norm)
plt.title("Distribución Precios",size=15, weight='bold')
plt.show()

get_ipython().set_next_input('En este caso que buscamos predecir el precio nos interesan las variables altamente correladas o por el contrario nos interesa descartarlas');get_ipython().run_line_magic('pinfo', 'descartarlas')


# La asimetria aqui indica que la variable se acerca bastante a una distribución normal, existe sesgo aun que no parece muy relevante 

# In[58]:


print("Skewness: %f" % df['price'].skew())
print("Kurtosis: %f" % df['price'].kurt())


# In[59]:


df['price'].describe()


# In[60]:


fig = plt.figure(figsize=(16,6))

ax1 = fig.add_subplot(121)
sns.boxplot(y = df['price'], ax=ax1, color='magenta')
describe = df['price'].describe().to_frame().round(2)

ax2 = fig.add_subplot(122)
ax2.axis('off')
font_size = 16
bbox = [0, 0, 1, 1]
table = ax2.table(cellText = describe.values, rowLabels = describe.index, bbox=bbox, colLabels=describe.columns)
table.set_fontsize(font_size)
fig.suptitle('Distribución de Precios', fontsize=16)
plt.show()


# El gráfico de probabilidad indica una normalidad aproximada.

# In[61]:


plt.figure(figsize=(7,7))
stats.probplot(df['price'], plot=plt)
plt.show()


# In[62]:


df['price'].value_counts()


# Con los datos relativamente limpios y ordenados sacaremos una matriz de correlación, utilizaremos la de Spearman ya que no está centrada en la distribución de las variables

# In[63]:


plt.figure(figsize=(15,12))
palette = sns.diverging_palette(20, 220, n=256)
corr=df.corr(method='spearman')
sns.heatmap(corr, annot=True, fmt=".2f", cmap=palette, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}).set(ylim=(13, 0))
plt.title("Matriz de Correlación",size=15, weight='bold')
plt.show()


# In[64]:


df_corr = df.corr(method='spearman') 
df_corr 


# In[65]:


corr = df.corr()
corr[['price']].sort_values(by = 'price',ascending = False).style.background_gradient()


# Según estos resultados las variables mas correladas con price son neighbourhood y calculated_host:listings_count, como correlación negativa podemos ver a room_type (cosa que no termino de entender)
# Por lo general, aparentemente, no existen variables que muestren gran influencia, ademas de algunas que parecen prescindibles. Seguiremos trabajando...

# In[66]:


df_x, df_y = df.iloc[:,:-1], df.iloc[:,-1]


# ###  Revision de sesgo en las caracteristicas

# Se puede ver que salvo price que antes tratamos las demas se muestran muy sesgadas. Sería recomendable tratarlas tambien?

# In[67]:


dists = df[['price', 'minimum_nights',
       'calculated_host_listings_count','number_of_reviews','reviews_per_month','availability_365']]

skewed_features = []
for column in dists:
    skew = abs(df[column].skew())
    print('{:15}'.format(column), 
          'Skewness: {:05.2f}'.format(skew),'Min value: {}'.format(df[column].min()))


# In[68]:


df['price'],price_lambda = boxcox(df['price'])
print('"price lambda": {}'.format(price_lambda))


# In[69]:


fig = plt.figure(figsize=(16,6))

ax1 = fig.add_subplot(121)
sns.boxplot(y = df['price'], ax=ax1, color='yellow')
describe = df['price'].describe().to_frame().round(2)

ax2 = fig.add_subplot(122)
ax2.axis('off')
font_size = 16
bbox = [0, 0, 1, 1]
table = ax2.table(cellText = describe.values, rowLabels = describe.index, bbox=bbox, colLabels=describe.columns)
table.set_fontsize(font_size)
fig.suptitle('Distribución de Precios', fontsize=16)
plt.show()


# In[70]:


fig = plt.figure(figsize=(16,6))

ax1 = fig.add_subplot(121)
sns.boxplot(y = df['minimum_nights'], ax=ax1, color='yellow')
describe = df['minimum_nights'].describe().to_frame().round(2)

ax2 = fig.add_subplot(122)
ax2.axis('off')
font_size = 16
bbox = [0, 0, 1, 1]
table = ax2.table(cellText = describe.values, rowLabels = describe.index, bbox=bbox, colLabels=describe.columns)
table.set_fontsize(font_size)
fig.suptitle('Distribución de minimum_nights', fontsize=16)
plt.show()


# In[71]:


plt.figure(figsize=(10,10))
sns.distplot(df['minimum_nights'], fit=stats.norm)
plt.title("Distribución minimum_nights",size=15, weight='bold')
plt.show()


# Se define la variable objetivo

# In[72]:


target = df['price']


# In[73]:


df.drop('price', axis=1, inplace=True)


# In[74]:


df_x, df_y = df.iloc[:,:-1], df.iloc[:,-1]


# In[77]:


from sklearn.model_selection import StratifiedShuffleSplit


# Realizo pruebas con este método de división aun que en este traabajo utilizo split
# 
# split = StratifiedShuffleSplit(n_splits=1, test_size=0.8, random_state=42)
# for train_index, test_index in split.split(df, df["room_type"]):
#     strat_train_set = df.loc[train_index]
#     strat_test_set = df.loc[test_index]
#     
# X_train = strat_train_set.drop('room_type',axis=1)
# y_train = strat_train_set["room_type"].copy()
# X_test = strat_test_set.drop('room_type',axis=1)
# y_test = strat_test_set["room_type"].copy()

# print('Dimensions of the training feature matrix: {}'.format(X_train.shape))
# print('Dimensions of the training target vector: {}'.format(y_train.shape))
# print('Dimensions of the test feature matrix: {}'.format(X_test.shape))
# print('Dimensions of the test target vector: {}'.format(y_test.shape))
# 

# Se dividen los datos en training y test (70%, 30%)

# In[74]:


X_train, X_test, y_train, y_test = train_test_split(df, target, test_size = .2, random_state=42)


# Tamaño de las particiones

# In[75]:


print('Dimensions of the training feature matrix: {}'.format(X_train.shape))
print('Dimensions of the training target vector: {}'.format(y_train.shape))
print('Dimensions of the test feature matrix: {}'.format(X_test.shape))
print('Dimensions of the test target vector: {}'.format(y_test.shape))
Está chulo
Gracias y Buenas Noches


# Ahora utilizaremos el clasificador ExtraTreesClasssifier de Scikit-Learn por su rapidez, con un estimador de 100 se puede
# observar el peso de cada característica

# In[77]:


unic = preprocessing.LabelEncoder()

nuevo_mod = ExtraTreesClassifier(n_estimators=60)
nuevo_mod.fit(X_train,unic.fit_transform(y_train))

plt.figure(figsize=(7,7))
caract = pd.Series(nuevo_mod.feature_importances_, index=df.iloc[:,:12].columns)
caract.nlargest(10).plot(kind='barh')
plt.show()


# ###  Validación Cruzada

# In[78]:


n_folds = 5

def rmse_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state = 64).get_n_splits(df)
    return cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=kf)


# In[95]:


from sklearn.preprocessing import PolynomialFeatures


# Opto por no utilizar la transformación polinomial añadiendo potencias a cada caracteristica y creando nuevas para tratar datos no lineales ya que tenia algun conflicto con Lime para visualizar mejor los resultados, con lo cual me hé decantado por lime y eli5.
# 
# Poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
# X_train = Poly.fit_transform(X_train)
# X_test = Poly.fit_transform(X_test)

# In[79]:


for LinearModel in [LinearRegression, RandomForestRegressor, XGBRegressor]:
    if LinearModel == XGBRegressor: 
        reg = rmse_cv(XGBRegressor(objective='reg:squarederror'))
    else: 
        reg = rmse_cv(LinearModel())
    print('{}: {:.3f} +/- {:.3f}'.format(LinearModel.__name__, -reg.mean(), reg.std()))


# In[ ]:


LinearRegression: 0.281 +/- 0.007
RandomForestRegressor: 0.250 +/- 0.012
XGBRegressor: 0.226 +/- 0.008


# LinearRegression: 0.027 +/- 0.000
# RandomForestRegressor: 0.022 +/- 0.000
# XGBRegressor: 0.019 +/- 0.000

# ### Definición de métricas del modelo

# RMSE (Raiz de la desviación cuadrática media): error cuadrático=(real − estimado)2. Indica cuanto se ajusta nuestro modelo a los datos, una aproximación a 0 sería idónea;
# 
# R^2: En rasgos genereales un indice alto significaria que el modelo se ajusta a los datos y su variabilidad de respuesta;
# 
# CV Error o MAE: Error que comete el modelo en la predicción;
# 
# CV Std - Desviación Típica o Estándar: Un valor bajo indicaria indicaria que la mayoria de los datos están cerca de la media.

# In[81]:


def rmse(actual,predicted):
    return(np.sqrt(mean_squared_error(actual, predicted)))


# In[82]:


def model_scores(df, cv_model, y_train, y_test, pred_train, pred_test):
    mse = mean_squared_error(y_test, lin_reg_pred_test)
    results = pd.DataFrame({'df':['{}'.format(type(df).__name__)],
                'CV error': '{:.3f}'.format(cv_model.mean()), 
                'CV std': '{:.3f}'.format(cv_model.std()),
                'RMSE train': [rmse(y_train, pred_train)],
                'RMSE test': [rmse(y_test, pred_test)],
                'R2 train': [r2_score(y_train, pred_train)],
                'R2 test': [r2_score(y_test, pred_test)]})
    results = results.round(decimals=4)

    return results


# In[83]:


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# Utilizaremos LIME ( Local Interpretable Model-agnostic Explanations ) para interpretar con mayor facilidad los resultados 
# de los modelos implementados

# In[84]:


explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=df.columns, class_names=['price'], verbose=True, mode='regression')
i = 500


# ###  Regreción Lineal

# In[85]:


lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

lin_reg_cv = -rmse_cv(LinearRegression())
lin_reg_pred_train = lin_reg.predict(X_train)
lin_reg_pred_test = lin_reg.predict(X_test)

lin_reg_results = model_scores(lin_reg, lin_reg_cv, y_train, y_test, lin_reg_pred_train, lin_reg_pred_test)
lin_reg_results.style.hide_index()


# In[87]:


lin_reg.coef_


# In[89]:


eli5.explain_weights(lin_reg, feature_names=list(df.iloc[:,:12].columns))  


# In[91]:


exp = explainer.explain_instance(X_test[i], lin_reg.predict, num_features=5)


# ###  Random Forest

# In[99]:


rfr_reg = RandomForestRegressor(random_state=42)
rfr_reg.fit(X_train, y_train) 

rfr_reg_cv = -rmse_cv(RandomForestRegressor(random_state = 3))
rfr_pred_train = rfr_reg.predict(X_train)
rfr_pred_test = rfr_reg.predict(X_test)

rfr_reg_results = model_scores(rfr_reg, rfr_reg_cv, y_train, y_test, rfr_pred_train, rfr_pred_test)

rfr_reg_results.style.hide_index()


# In[101]:


eli5.show_weights(rfr_reg, feature_names=list(df.iloc[:,:12].columns))


# In[102]:


exp = explainer.explain_instance(X_test[i], rfr_reg.predict, num_features=5)


# ###  XGBoost

# In[104]:


xgb_reg = XGBRegressor(n_estimators=1000)
xgb_reg.fit(X_train, y_train, early_stopping_rounds=10, # si en 10 iteraciones no hay mejora que se detenga
             eval_set=[(X_test, y_test)], verbose=False)


# In[105]:


xgb_reg = XGBRegressor(learning_rate=0.3,
                      n_estimators=1000, 
                      max_depth=5, min_child_weight=1) # max depth por defecto 6, lo elevaré, seguramente solo consiga
                                                       # overfitting
xgb_reg_cv = -rmse_cv(xgb_reg)

xgb_reg.fit(X_train, y_train)
xgb_pred_train = xgb_reg.predict(X_train)
xgb_pred_test = xgb_reg.predict(X_test)

xgb_reg_results1 = model_scores(xgb_reg, xgb_reg_cv, y_train, y_test, xgb_pred_train, xgb_pred_test)

xgb_reg_results1.style.hide_index()


# In[ ]:


# Profundidad = 8
df	CV error	CV std	RMSE train	RMSE test	R2 train	R2 test
XGBRegressor	0.021	0.001	0.0164	0.1431	0.995	0.6155


# In[ ]:


# Profundidad = 10
df	CV error	CV std	RMSE train	RMSE test	R2 train	R2 test
XGBRegressor	0.021	0.000	0.0027	0.143	0.9999	0.616


# In[ ]:


# Profundidad = 6
XGBRegressor	0.021	0.000	0.0508	0.1412	0.9522	0.6256


# In[106]:


eli5.show_weights(xgb_reg, feature_names=list(df.iloc[:,:12].columns))


# In[107]:


exp = explainer.explain_instance(X_test[i], xgb_reg.predict, num_features=5)


# Como en la parte superior realizamos una transformación Boxcox ahora tenemos que hacer una inversión, para poder
# interpretar los resultados.

# In[109]:


y_test = inv_boxcox(y_test, price_lambda)
y_train = inv_boxcox(y_train, price_lambda)


# In[110]:


rfr_pred_train = inv_boxcox(rfr_pred_train, price_lambda)
rfr_pred_test = inv_boxcox(rfr_pred_test, price_lambda)


# Creamos un Data Frame con los modelos para facilitar la comparación de los resultados

# In[111]:


results = pd.DataFrame({
                'MSE train': [mean_squared_error(y_train, rfr_pred_train)],
                'MSE test': [mean_squared_error(y_test, rfr_pred_test)],
                'RMSE train': [np.sqrt(mean_squared_error(y_train, rfr_pred_train))],
                'RMSE test': [np.sqrt(mean_squared_error(y_test, rfr_pred_test))],
                'R2 train': [r2_score(y_train, rfr_pred_train)],
                'R2 test': [r2_score(y_test, rfr_pred_test)]})
results = results.round(decimals=4)
results


# In[112]:


xgb_pred_train = inv_boxcox(xgb_pred_train, price_lambda)
xgb_pred_test = inv_boxcox(xgb_pred_test, price_lambda)


# In[113]:


results = pd.DataFrame({
                'MSE train': [mean_squared_error(y_train, xgb_pred_train)],
                'MSE test': [mean_squared_error(y_test, xgb_pred_test)],
                'RMSE train': [np.sqrt(mean_squared_error(y_train, xgb_pred_train))],
                'RMSE test': [np.sqrt(mean_squared_error(y_test, xgb_pred_test))],
                'R2 train': [r2_score(y_train, xgb_pred_train)],
                'R2 test': [r2_score(y_test, xgb_pred_test)]})
results = results.round(decimals=4)
results


# In[114]:


lin_reg_pred_train = inv_boxcox(lin_reg_pred_train, price_lambda)
lin_reg_pred_test = inv_boxcox(lin_reg_pred_test, price_lambda)


# In[115]:


results = pd.DataFrame({
                'MSE train': [mean_squared_error(y_train, lin_reg_pred_train)],
                'MSE test': [mean_squared_error(y_test, lin_reg_pred_test)],
                'RMSE train': [np.sqrt(mean_squared_error(y_train, lin_reg_pred_train))],
                'RMSE test': [np.sqrt(mean_squared_error(y_test, lin_reg_pred_test))],
                'R2 train': [r2_score(y_train, lin_reg_pred_train)],
                'R2 test': [r2_score(y_test, lin_reg_pred_test)]})
results = results.round(decimals=4)
results


# A seguir se muestran en un dataframe los valores obtenidos para cada modelo

# In[116]:


error = pd.DataFrame({'Valores Reales': np.array(y_test).flatten(), 'Bosque Aleatório': rfr_pred_test.flatten(),
                      'XGBoost': xgb_pred_test.flatten(), 'Regresión Lineal': lin_reg_pred_test.flatten()})
error.head(10)


# ###  Gráfico con la predicción de XGBoost

# In[117]:


plt.figure(figsize=(15,7))
sns.regplot(y=xgb_pred_test, x=y_test, line_kws={"color": "red"}, color='skyblue')
plt.title('Evalución de Predicciones', fontsize=15)
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.show()


# Repetiremos los mismos pasos aun que en esta ocasion sin algunas cartacteristicas

# In[118]:


df.info()


# In[119]:


df = df.drop(columns=['name','host_id','neighbourhood_group', 'calculated_host_listings_count'])
df.isnull().sum()


# In[120]:


df_x, df_y = df.iloc[:,:-1], df.iloc[:,-1]


# In[121]:


X_train, X_test, y_train, y_test = train_test_split(df, target, test_size = .2, random_state=42)


# In[122]:


print('Dimensions of the training feature matrix: {}'.format(X_train.shape))
print('Dimensions of the training target vector: {}'.format(y_train.shape))
print('Dimensions of the test feature matrix: {}'.format(X_test.shape))
print('Dimensions of the test target vector: {}'.format(y_test.shape))


# In[123]:


unic = preprocessing.LabelEncoder()

nuevo_mod = ExtraTreesClassifier(n_estimators=60)
nuevo_mod.fit(X_train,unic.fit_transform(y_train))

plt.figure(figsize=(7,7))
caract = pd.Series(nuevo_mod.feature_importances_, index=df.iloc[:,:14].columns)
caract.nlargest(10).plot(kind='barh')
plt.show()


# In[124]:


n_folds = 5

def rmse_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state = 64).get_n_splits(df)
    return cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=kf)


# In[125]:


for LinearModel in [LinearRegression, RandomForestRegressor, XGBRegressor]:
    if LinearModel == XGBRegressor: 
        reg = rmse_cv(XGBRegressor(objective='reg:squarederror'))
    else: 
        reg = rmse_cv(LinearModel())
    print('{}: {:.3f} +/- {:.3f}'.format(LinearModel.__name__, -reg.mean(), reg.std()))


# In[126]:


def rmse(actual,predicted):
    return(np.sqrt(mean_squared_error(actual, predicted)))


# In[127]:


def model_scores(df, cv_model, y_train, y_test, pred_train, pred_test):
    mse = mean_squared_error(y_test, lin_reg_pred_test)
    results = pd.DataFrame({'df':['{}'.format(type(df).__name__)],
                'CV error': '{:.3f}'.format(cv_model.mean()), 
                'CV std': '{:.3f}'.format(cv_model.std()),
                'RMSE train': [rmse(y_train, pred_train)],
                'RMSE test': [rmse(y_test, pred_test)],
                'R2 train': [r2_score(y_train, pred_train)],
                'R2 test': [r2_score(y_test, pred_test)]})
    results = results.round(decimals=4)

    return results


# In[128]:


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# In[129]:


explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=df.columns, class_names=['price'], verbose=True, mode='regression')
i = 500


# Regreción Lineal

# In[130]:


lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

lin_reg_cv = -rmse_cv(LinearRegression())
lin_reg_pred_train = lin_reg.predict(X_train)
lin_reg_pred_test = lin_reg.predict(X_test)

lin_reg_results = model_scores(lin_reg, lin_reg_cv, y_train, y_test, lin_reg_pred_train, lin_reg_pred_test)
lin_reg_results.style.hide_index()


# In[131]:


lin_reg.coef_


# In[132]:


eli5.explain_weights(lin_reg, feature_names=list(df.columns))


# In[133]:


exp = explainer.explain_instance(X_test[i], lin_reg.predict, num_features=5)


# ###  Random Forest

# In[135]:


rfr_reg = RandomForestRegressor(random_state=42)
rfr_reg.fit(X_train, y_train) 

rfr_reg_cv = -rmse_cv(RandomForestRegressor())
rfr_pred_train = rfr_reg.predict(X_train)
rfr_pred_test = rfr_reg.predict(X_test)

rfr_reg_results = model_scores(rfr_reg, rfr_reg_cv, y_train, y_test, rfr_pred_train, rfr_pred_test)

rfr_reg_results.style.hide_index()


# In[136]:


eli5.show_weights(rfr_reg, feature_names=list(df.columns))


# In[137]:


exp = explainer.explain_instance(X_test[i], rfr_reg.predict, num_features=5)


# ###  XGBoost

# In[139]:


xgb_reg = XGBRegressor(n_estimators=1000)
xgb_reg.fit(X_train, y_train, early_stopping_rounds=10, 
             eval_set=[(X_test, y_test)], verbose=False)


# In[140]:


xgb_reg = XGBRegressor(learning_rate=0.3,
                      n_estimators=1000,
                      max_depth=5, min_child_weight=1)

xgb_reg_cv = -rmse_cv(xgb_reg)

xgb_reg.fit(X_train, y_train)
xgb_pred_train = xgb_reg.predict(X_train)
xgb_pred_test = xgb_reg.predict(X_test)

xgb_reg_results1 = model_scores(xgb_reg, xgb_reg_cv, y_train, y_test, xgb_pred_train, xgb_pred_test)

xgb_reg_results1.style.hide_index()


# In[141]:


eli5.show_weights(xgb_reg, feature_names=list(df.columns))


# In[142]:


exp = explainer.explain_instance(X_test[i], xgb_reg.predict, num_features=5)


# ###  Inversión para la interpretación

# In[144]:


y_test = inv_boxcox(y_test, price_lambda)
y_train = inv_boxcox(y_train, price_lambda)


# In[145]:


rfr_pred_train = inv_boxcox(rfr_pred_train, price_lambda)
rfr_pred_test = inv_boxcox(rfr_pred_test, price_lambda)


# In[146]:


results = pd.DataFrame({
                'MSE train': [mean_squared_error(y_train, rfr_pred_train)],
                'MSE test': [mean_squared_error(y_test, rfr_pred_test)],
                'RMSE train': [np.sqrt(mean_squared_error(y_train, rfr_pred_train))],
                'RMSE test': [np.sqrt(mean_squared_error(y_test, rfr_pred_test))],
                'R2 train': [r2_score(y_train, rfr_pred_train)],
                'R2 test': [r2_score(y_test, rfr_pred_test)]})
results = results.round(decimals=4)
results


# In[147]:


xgb_pred_train = inv_boxcox(xgb_pred_train, price_lambda)
xgb_pred_test = inv_boxcox(xgb_pred_test, price_lambda)


# In[148]:


results = pd.DataFrame({
                'MSE train': [mean_squared_error(y_train, xgb_pred_train)],
                'MSE test': [mean_squared_error(y_test, xgb_pred_test)],
                'RMSE train': [np.sqrt(mean_squared_error(y_train, xgb_pred_train))],
                'RMSE test': [np.sqrt(mean_squared_error(y_test, xgb_pred_test))],
                'R2 train': [r2_score(y_train, xgb_pred_train)],
                'R2 test': [r2_score(y_test, xgb_pred_test)]})
results = results.round(decimals=4)
results


# In[149]:


lin_reg_pred_train = inv_boxcox(lin_reg_pred_train, price_lambda)
lin_reg_pred_test = inv_boxcox(lin_reg_pred_test, price_lambda)


# In[150]:


results = pd.DataFrame({
                'MSE train': [mean_squared_error(y_train, lin_reg_pred_train)],
                'MSE test': [mean_squared_error(y_test, lin_reg_pred_test)],
                'RMSE train': [np.sqrt(mean_squared_error(y_train, lin_reg_pred_train))],
                'RMSE test': [np.sqrt(mean_squared_error(y_test, lin_reg_pred_test))],
                'R2 train': [r2_score(y_train, lin_reg_pred_train)],
                'R2 test': [r2_score(y_test, lin_reg_pred_test)]})
results = results.round(decimals=4)
results


# ###  Dataframe con los resultados de los modelos 

# In[151]:


error = pd.DataFrame({'Real Values': np.array(y_test).flatten(), 'Randon Forest': rfr_pred_test.flatten(),
                      'XGB': xgb_pred_test.flatten(), 'Regresión Lineal': lin_reg_pred_test.flatten()})
error.head(10)


# Tras la retirada de algunas características, el mejor mejor resultado lo vuelve a obtener xgboost

# In[152]:


plt.figure(figsize=(15,7))
sns.regplot(y=xgb_pred_test, x=y_test, line_kws={"color": "red"}, color='skyblue')
plt.title('Evalución de Predicciones', fontsize=15)
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.show()


# ### Conclusión

# Para poder visualizar y entender los datos se utilizan distintos gráficos y mapas, el tratamiento de variables que contenian datos faltantes, en algun caso se igualan a la media y en otros se eliminan del dataset, en el caso de la variable objetivo que contenia entradas iguales a 0 o superiores a 8000$ se opta por descartar dichos valores, teniendo en cuenta que eran mínimos.
# 
# Tambien se muestra matriz de correlación entre los datos numéricos, primero de Pearson y de Spearman tras tratar algunas variables con cat.code.
# 
# En el caso de price se trata con boxcox y se logra una aproximación a la normalidad.
# Para dividir del dataset se utiliza split aun que tambien se realiza una prueba con StratifiedShuffleSplit.
# Se realizan pruebas en dos etapas distintas con Random Forest, Regresión Lineal y XGBoost, se ajustan algunos parámetros para obtener mejores resultados.
# XGBoost muestra un mejor desempeño en las dos etapas.
# 
# Existen muchos pasos a mejorar para obtener reultados con menos errores, el caso de los hiper parámetros, el trato de las variables cuantitativas y cualitativas, las transformaciones, otros modelos, etc., distintos detalles que confío solventar con más tiempo de estudio.

# El objetivo del proyecto era poder poner en prática lo aprendido. Deseo indicar que durante el desarrollo de este, hé podido constatar mi evolución del princípio hasta ahora, unas fases me resultaron sencillas y muchas mas, muy complicadas, ademas de repetir las clases varias veces tambien me apoyé en el material facilitado, como es el caso de los libros y páginas. Leí y revisé trabajos publicados, adapté código de distintas fuentes y aprendí mucho en el proceso.
# Tengo un largo camino de trabajo aun en esta materia y durante este tiempo se han visto reforzadas las ganas y la voluntadad de seguir poniendo empeño en saber cada dia un poco más.
# 
# Aprovecho también la ocasión para poner en valor el trabajo y la atención de mi Profesora Maricarmen. Por la amabilidad y la calidez al transmitir coneceptos y conocimientos que no siempre resultan sencillos de explicar y hacer entender, Gracias.
# 

# In[ ]:




