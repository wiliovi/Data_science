import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression



#Descenso del gradiente

'''en este caso en cada iteraccion se calcula 'STEP_SIZE*(f_prima(x))'
por lo que el coste computacional es muy grande'''

f = lambda x: x**2 - 2*x + 4
f_prima = lambda x: 2*x - 2

STEP_SIZE = 0.02

def descenso_gradiente(x):
    return x - STEP_SIZE*(f_prima(x))

minimo_iteraciones = []
N_ITERACIONES = 100
x = 3

for i in range(N_ITERACIONES):
    minimo_iteraciones.append(x)
    x = descenso_gradiente(x)

#Batch gradient descent
'''para solucionar el problema computacional se escoge para cada iteraccion una muestra aleatoria
'STEP_SIZE*(f_prima(x_aleatoria))' por lo que el coste computacional es menor en cada iteraccion'''

n_muestras = 1000
n_variables = 2

X,y,coeficientes_objetivo = make_regression(n_samples=n_muestras,
                        n_features=n_variables,
                        coef=True)

'''Podemos obtener la variable objetivo mediante un producto escalar de los pesos y sus variables independientes '''
def predecir_batch(coeficientes, X):
    return coeficientes @ X.T
'''Necesitamos una funcion de error, en este caso usaremos el Error cuadratico medio '''
def error_batch(y_pred, y_true):
    m = y_pred.shape[0]
    return (np.sum(y_pred - y_true)**2)/2*m
'''Tambien necesitamos la derivada de la funcion de Error'''
def derivada_error_batch(y_pred, y_true, x):
    m = y_pred.shape[0]
    return np.sum((y_pred - y_true)*x/m)
'''Definimos la funcion principal'''
def descenso_gradiente_batch(coeficientes, X, y):
    y_predicciones = predecir_batch(coeficientes, X)
    for i in range(coeficientes.shape[0]):
        coeficientes[i] = coeficientes[i]- STEP_SIZE * derivada_error_batch(y_predicciones, y, X[:,i])
    error = error_batch(y_predicciones, y)
    return coeficientes, error

coeficientes_iteraciones = []
error_iteraciones = []

N_ITERACIONES = 200
STEP_SIZE = 0.02
coeficientes = np.random.random((X.shape[1],))
error = error_batch(coeficientes, X)

for i in range(N_ITERACIONES):
    coeficientes_iteraciones.append(coeficientes.copy())
    error_iteraciones.append(error)
    coeficientes, error = descenso_gradiente_batch(coeficientes, X, y)

coeficientes_iteraciones = np.array(coeficientes_iteraciones)

print(coeficientes)

plt.figure()

plt.subplot(2,2,1)
plt.plot(error_iteraciones)
plt.title("Evolución del error con el número de iteraciones")

plt.subplot(2,2,3)
plt.plot(coeficientes_iteraciones[:,0], color="red")
plt.axhline(coeficientes_objetivo[0], color="red", linestyle="dashed")

plt.plot(coeficientes_iteraciones[:,1], color="blue")
plt.axhline(coeficientes_objetivo[1], color="blue", linestyle="dashed")

plt.xlabel("Numero de iteraciones")
plt.ylabel("Valor del coeficiente")
plt.title("Evolución de coeficientes con el número de iteraciones")

plt.subplot(2,2,2)
plt.plot(minimo_iteraciones)
plt.title("Evolución del error")

plt.show()
