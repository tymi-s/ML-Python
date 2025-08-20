from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', None)


url= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/pu9kbeSaAtRZ7RxdJKX9_A/yellow-tripdata.csv'
raw_data = pd.read_csv(url)
print(raw_data.head())


# zrozumienie zbioru i korelacja:

correlation_values= raw_data.corr()['tip_amount'].drop('tip_amount')
correlation_values.plot(kind='barh')
plt.title('Korelacja')
plt.show()


######################preporcessing
# normalizacja:
y = raw_data[['tip_amount']].values.astype('float32')

# usuwanie kolumny ze zmienną docelową
proc_data = raw_data.drop(['tip_amount'],axis=1)

#zmienne zależne:

X = proc_data.values# .values konwertuje wartości na macierze numeryczne do treningu i obliczeń
X = normalize(X, axis=1, norm='l1', copy=False)# normalizacja


# podział danych na treningowe i testowe
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

################### Budowanie Regresora Decision Tree:
from sklearn.tree import DecisionTreeRegressor

dt_reg = DecisionTreeRegressor(criterion='squared_error', max_depth=5,random_state=35)

#trening
dt_reg.fit(X_train, y_train)


#predykcja:
y_pred = dt_reg.predict(X_test)

#błąd średniokwadratowy na danych testowych
mse_score = mean_squared_error(y_test, y_pred)
print('MSE score : {0:.3f}'.format(mse_score))


r2_score = dt_reg.score(X_test,y_test)
print('R^2 score : {0:.3f}'.format(r2_score))


