#Multiple Linear Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
######################################################################################################################################################################################
url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"

df = pd.read_csv(url)
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', None)
#print(df.sample(5))
#print(df.describe())
#print(df.head())
######################################################################################################################################################################################
#usunięcie kolumny MODELYEAR bo wsystkie są takie same dla każdego autka:
df = df.drop(['MODELYEAR', 'MAKE', 'MODEL', 'VEHICLECLASS', 'TRANSMISSION', 'FUELTYPE',],axis=1)

##################################################################################################################################################################################################################
#macierz korelacji aby sprawdzić zależność między zmiennnymi. Celem jest eliminacja silnych zależności i korelacji między zmiennymi żeby nie były redandentne.

print(df.corr())
# korelacja z docelową zmienną jest bardzo duże ze wszystkimi innymi bo wszedzie przekreacza 85 % (ostatni wiersz)

# analiza macierzy korelacji:
# ENGINESIZE jest bardziej skorelowane z targetem niż CYLINDERS więc CYLINDERS można dropnąć:
# jesli chodzi o reszte to FUELCONSUMPTION_COMB_MPG jest najbardziej skorelowane z targetem więc reszte można dropnąć. oznacza to że zostaną dwie zmienne ENGINESIZE i FUELCONSUMPTION_COMB_MPG.
df = df.drop(['CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB'],axis=1)
print(df.head(2))




##################################################################################################################################################################################################################

# macierz rozrzutut żeby pomóc w wybraniu cech które nie są zbędne:
axes = pd.plotting.scatter_matrix(df, alpha=0.2)

#obrót żeby było je można odczytać:

for ax in axes.flatten():
    ax.xaxis.label.set_rotation(90)
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha('right')

#ploting:
plt.tight_layout()
plt.gcf().subplots_adjust(wspace=0, hspace=0)
#plt.show()

# wniosek: zależność między FUELCONSUMPTION_COMB_MPG a CO2 jest nieliniowa.


###################################################################################################################################################################################################################
#PRZYGOTOWANIE DANYCH DO WIELOKROTNEJ REGRESJI LINIOWEJ:

X = df.iloc[:,[0,1]].to_numpy()
y = df.iloc[:,[2]].to_numpy()# konwersja do tablicy numpy


###################################################################################################################################################################################################################
#preprocessing - przekształcenie X tak, aby miały średnią 0 i odchylenie standardowe 1 (ważne w wielu modelach ML).

from sklearn import preprocessing
std_scaler = preprocessing.StandardScaler()
X_std= std_scaler.fit_transform(X)
pd.DataFrame(X_std).describe().round(2)

###################################################################################################################################################################################################################
#podział danych na te do treningu i testów w proporcji 80/20

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

###################################################################################################################################################################################################################
###################################################################################################################################################################################################################
#MODEL WIELOKROTNEJ REGRESJI LINIOWEJ:

from sklearn.linear_model import LinearRegression
#model object
regressor = LinearRegression()

#training a model
regressor.fit(X_train,y_train)

#wypisanie współczynników:
coef = regressor.coef_
intercept= regressor.intercept_

print ('Coefficients: ',coef)
print ('Intercept: ',intercept)

#Parametry współczynników (coef) i przecięcia (intercept) definiują najlepiej dopasowaną hiperpłaszczyznę do danych
#Ponieważ istnieją tylko dwie zmienne, a zatem dwa parametry, hiperpłaszczyzna jest płaszczyzną.
# Jednak ta najlepiej dopasowana płaszczyzna będzie wyglądać inaczej w oryginalnej, niestandaryzowanej przestrzeni cech.

###################################################################################################################################################################################################################
# teraz te dane wyjściowe trzeba przekształcić do postaci takiej jakie były dane wejściowe aby zrozumieć ich sens. Bez tego etapu całą ta regresja jest abstrakcyjna i bez senus.

# Get the standard scaler's mean and standard deviation parameters
means_ = std_scaler.mean_
std_devs_ = np.sqrt(std_scaler.var_)

# The least squares parameters can be calculated relative to the original, unstandardized feature space as:
coef_original = coef / std_devs_
intercept_original = intercept - np.sum((means_ * coef) / std_devs_)

print ('Coefficients po przekształceniu ', coef_original)
print ('Intercept po przekształceniu : ', intercept_original)


###################################################################################################################################################################################################################
###################################################################################################################################################################################################################

#ploting:

#from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# Ensure X1, X2, and y_test have compatible shapes for 3D plotting
X1 = X_test[:, 0] if X_test.ndim > 1 else X_test
X2 = X_test[:, 1] if X_test.ndim > 1 else np.zeros_like(X1)

# Create a mesh grid for plotting the regression plane
x1_surf, x2_surf = np.meshgrid(np.linspace(X1.min(), X1.max(), 100),
                               np.linspace(X2.min(), X2.max(), 100))

y_surf = intercept +  coef[0,0] * x1_surf  +  coef[0,1] * x2_surf

# Predict y values using trained regression model to compare with actual y_test for above/below plane colors
y_pred = regressor.predict(X_test.reshape(-1, 1)) if X_test.ndim == 1 else regressor.predict(X_test)
above_plane = y_test >= y_pred
below_plane = y_test < y_pred
above_plane = above_plane[:,0]
below_plane = below_plane[:,0]

# Plotting
fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the data points above and below the plane in different colors
ax.scatter(X1[above_plane], X2[above_plane], y_test[above_plane],  label="Above Plane",s=70,alpha=.7,ec='k')
ax.scatter(X1[below_plane], X2[below_plane], y_test[below_plane],  label="Below Plane",s=50,alpha=.3,ec='k')

# Plot the regression plane
ax.plot_surface(x1_surf, x2_surf, y_surf, color='k', alpha=0.21,label='plane')

# Set view and labels
ax.view_init(elev=10)

ax.legend(fontsize='x-large',loc='upper center')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect(None, zoom=0.75)
ax.set_xlabel('ENGINESIZE', fontsize='xx-large')
ax.set_ylabel('FUELCONSUMPTION', fontsize='xx-large')
ax.set_zlabel('CO2 Emissions', fontsize='xx-large')
ax.set_title('Multiple Linear Regression of CO2 Emissions', fontsize='xx-large')
plt.tight_layout()
plt.show()
