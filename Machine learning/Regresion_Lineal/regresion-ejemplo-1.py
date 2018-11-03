import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

X = [[2],[4],[5],[6],[8]]
Y = [[5],[9],[11],[13],[17]]

X1 = [[3],[5],[6],[7],[9]]


print(zip(X,Y))

'''Modelo regresion lineal'''
lr = LinearRegression()
lr.fit(X = X, y = Y)
print('Ecuacion de la recta')
print('Y = {} + {} . X1'.format(lr.intercept_,lr.coef_))

Y1 = lr.predict(X1)
print(Y1)

'''Diagrama de dispersion'''
plt.title('Diagrama de dispersion')
plt.scatter(X,Y,c='red',label='num')
plt.scatter(X1,Y1,c='blue',label='predict')
plt.plot(X1,Y1,color='black',linewidth=3)
plt.legend(loc='upper right')
plt.show()
