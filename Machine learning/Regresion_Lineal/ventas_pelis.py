import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

pelis = pd.read_csv('movies.csv')

'''Visualizacion de datos categoricos'''

plt.figure()

#Grafico de barras
plt.subplot(1,2,1)
(100 * pelis['genero'].value_counts() / len(pelis['genero'])).plot(kind='bar',title='Genero de peliculas %')

#Grafico de tartas
plt.subplot(1,2,2)
pelis['pais'].value_counts().plot(kind='pie', autopct='%.2f', figsize=(6, 6),title='Pais de origen')

#Relacion de variables categóricas
'''Vamos a analizar que relacion existe entre el pais y el genero'''

print(pd.crosstab(index=pelis['pais'],columns=pelis['genero'],margins=True).head(4))
pd.crosstab(index=pelis['pais'],columns=pelis['genero']).apply(lambda r: r/r.sum() *100,axis=1).plot(kind='bar')

#Relacion entre una variable categorica y una cuantitativa
genero = pelis.groupby('genero')['ventas'].mean()
genero.head(10).plot.barh()

plt.show()

'''Dentro de nuestro modelo solo usaremos los datos numericos'''
pelis = pelis.select_dtypes(include=['float64']).fillna(0)
print(pelis.head(5))

'''Usaremos el modelo de regresion lineal para estimar unposible resultado'''
var_o = 'ventas'
'''var_i = pelis.drop(columns=var_o).columns'''
var_i = ['presupuesto','popularidad']

objetivo =  pelis[var_o].values
datos =  pelis[var_i].values

lr = LinearRegression()
lr.fit(X = datos, y = objetivo)

pelis['Prediccion de ventas']=lr.predict(datos)
print(pelis[['ventas','Prediccion de ventas']].head(5))
