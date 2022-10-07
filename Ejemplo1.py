import pandas as pd
import numpy as np

data = pd.read_excel('BI_Alumnos07.xlsx')
from sklearn import linear_model

#Prueba de indicadores
from sklearn.metrics import mean_squared_error, r2_score

print(data.shape)
print(data.describe())

VIndependiente = data[['Altura']]
x = np.array(VIndependiente)

VDependeiente = data['Peso'].values

#Metodo de regresión lineal
RL = linear_model.LinearRegression()
RL.fit(x, VDependeiente)

#Saber si hay relacion entre variables + 0.5
I_Pred = RL.predict(x)
print()
print('Coeficiencde de R: ', RL.coef_)
print()

#Tiene que acercarse a 100 para ser valida
print('Termino Independiente: ', RL.intercept_)
print()
#Cada vez se debe acercar a 0 / margen de error
print('Error de cuadradado medio: %.2f' %  mean_squared_error(VDependeiente,I_Pred))
print()
#Se debe acercar a 1
print('Puntaje de varianza: %.2f ' % r2_score(VDependeiente, I_Pred))
P_Peso = RL.predict([[180]])
print()
#Predicción
print('Prediccion de peso: ', int(P_Peso))