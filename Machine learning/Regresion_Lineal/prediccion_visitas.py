import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12,12)
plt.rcParams['font.size'] = 16

from sklearn.linear_model import LinearRegression

art = pd.read_csv('articulos_ml.csv')

variable_independiente = ['Word count']
variable_dependiente = '# Shares'

X = art[variable_independiente].as_matrix()
Y = art[variable_dependiente].as_matrix()
X_T = X.T
betas = np.linalg.inv(X_T @ X) @ X_T @ Y
alfa = Y.mean() - np.dot(betas,art[variable_independiente].mean().as_matrix())

def predecir(r):
    return alfa + np.dot(betas, r.values)

def error_cuadrático_medio(y, y_pred):
    return np.sum((y-y_pred)**2)/len(y)

art['Share_pred'] = art[variable_independiente].apply(predecir,axis=1)
error_training = error_cuadrático_medio(art['# Shares'].values, art.Share_pred)

print(error_training)
print(art[['# Shares','Share_pred']].head())

plt.scatter(art['Word count'].values,art['# Shares'].values, alpha=0.5, label="real")
plt.plot(art['Word count'].values,art.Share_pred, c="black", label="prediccion")
plt.legend()
plt.show()
