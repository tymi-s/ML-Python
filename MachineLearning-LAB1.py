#Simple Linear Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df = pd.read_csv(url)# zapisanie danych do zmiennej

print("SAMPLE\n",df.sample(5))# wypisanie randomowych pięciu wierszy
print("\nDESCRIBE\n" ,df.describe())# dane statystyczne

#wypisanie tylko niektórych kolumn:

cdf = df[['CYLINDERS','ENGINESIZE','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print("\nNIEKTÓRE KOLUMNY\n",cdf.sample(10))


#rysowanie histogramów
bdf = df[['CYLINDERS','ENGINESIZE','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
bdf.hist()
plt.show()
#wniosek - fuel consumption jest powiązane z CO2 emissions bo historgramy sa podobne


# rysowanie scatter-plotów żeby sprawdzić zalezność liniową tych dwuch zmiennych:

plt.scatter(bdf.FUELCONSUMPTION_COMB,bdf.CO2EMISSIONS,color = 'purple')
plt.xlabel('Fuel Consumption')
plt.ylabel('CO2EMISSIONS')
plt.grid(True)
plt.xlim(0,27)#skalowanie wartości na wykresie
plt.show()
#wniosek - są trzy grupy aut któe mają wyraźną zależność liniową


# scatter-plot między engine size i co2 emission:

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("CO2EMISSIONS")
plt.grid(True)
plt.xlim(0,27)
plt.show()


# sprawdzenie liniowej zaleznosći miedzy cylindrem a CO2 emission:
plt.scatter(df.CYLINDERS,df.CO2EMISSIONS,color='orange')
plt.xlabel("CYLINDERS")
plt.ylabel("CO2EMISSIONS")
plt.grid(True)
plt.xlim(0,27)
plt.show()

#########################################################################################################
# CO2 EMISSION BĘDZIE PRZEWIDYWANE NA PODSTAWIE ENGINE SIZE. Przygotowanie danych:

x=cdf.ENGINESIZE.to_numpy()
y=cdf.CO2EMISSIONS.to_numpy()

# teraz będzie dzieleniea danych na te do trenowania i te do testowania.
# Im więcej danych tym większy ich procent musi być przeznaczony na trenowanie modelu regresji

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)# 20 procent na trening
# rezultatem są jednowymiarowe tablice NumPy


##########################################################################################################
#BUDOWANIE PROSTEGO MODELU REGRESJI LINIOWEJ przy użyicu scikit-learn:

from sklearn import linear_model

# model object

regressor = linear_model.LinearRegression()

# trenowanie modelu na danych do treningu:
# X_train to tablica 1-D, ale modele sklearn oczekują tablicy 2D jako danych wejściowych dla danych treningowych o kształcie (n_obserwacji,n_featerów)
# Więc musimy go przekształcić. Możemy pozwolić mu wnioskować o liczbie obserwacji przy użyciu „-1”.
regressor.fit(X_train.reshape(-1,1),y_train)

# wypisanie współczynników (coeffirences):
print("coeffficients",regressor.coef_[0])# z racji że jest to regresja liniowa to jest tu tylko jeden współczynnik
print("intercept",regressor.intercept_)

#coefficients i intercept to parametry regresji zdeterminowane przez model.
# coefficients = a
# intercept = b
# w równaniu regresji ax+b


# wyświetlenie wyników:

plt.scatter(X_train,y_train,color='grey')
plt.plot(X_train,regressor.coef_*X_train+regressor.intercept_,color='brown')
plt.xlabel("Engine size")
plt.ylabel("CO2EMISSIONS")
plt.title("RESULTS OF LINEAR REGRESSION")
plt.show()



#########################################################################################
# końcowy etap czyli ocena modelu i obliczenie błędów:
# - średni błąd
# średnia kwadratowa błędu (MSE)
# RMSE
# R2-score - czyli jak dobzrze model przybliża wartość rzeczywistą. im większy tym lepszy, największa wartość to 1.0

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Use the predict method to make test predictions
y_test_ = regressor.predict(X_test.reshape(-1,1))

# Evaluation
print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_test_))
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_test_))
print("Root mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test, y_test_)))
print("R2-score: %.2f" % r2_score(y_test, y_test_))



###############wynik dokładny:
plt.scatter(X_test, y_test,  color='blue')
plt.plot(X_test, regressor.coef_ * X_test + regressor.intercept_, '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.title("EXACT RESULTS ")
plt.show()



