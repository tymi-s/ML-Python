import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

############################################################################
#pobranie danych

url =  "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
churn_df = pd.read_csv(url)
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', None)

print(churn_df.head())

# zmiana na inta:
churn_df['churn'] = churn_df['churn'].astype('int')

print( churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn']])

############################ przypisanie danych do zmiennych:

y= np.asarray(churn_df['churn'])
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip','callcard']])


# teraz warto zestandaryzować te dane aby wszystkie były w tej samej skali. Model wtedy uczy się szybciej i poprawia jego wydajność

X_norm = StandardScaler().fit(X).transform(X)

#podział danych na testowe i treningowe:

X_train,X_test,y_train,y_test=train_test_split(X_norm,y,test_size=0.2,random_state=4)


##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
#tworzenie modelu Regresji Logistycznej i jego trening

LR = LogisticRegression().fit(X_train,y_train) # fiting to to samo co trening

#przewidywanie zmiennnej zależnej:
yhat = LR.predict_proba(X_test)
print(yhat[:10])
yhat_output = LR.predict(X_test)
print(yhat_output[:10])

#skoro celem jest jak najdokładniejsze obliczenie pradowpodobieństwa można sprawdzić
# jaki udział w tym ma każda ze zmiennych niezależnych:

feature_names = ['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip','callcard']
coefficients = pd.Series(LR.coef_[0], index=feature_names)

coefficients.sort_values().plot(kind='barh')
plt.title("Feature Coefficients in Logistic Regression Churn Model")
plt.xlabel("Coefficient Value")
plt.show()


# duży pasek na wykresie dla danej zmiennej oznacza że zwiększenie tej wartości
#dla danego wiersza spowodowałoby zwiększenie prawdopodobieństwa
#na wystąpienie 1 w Churn


# obliczenie skali błedu log-loss:

print(log_loss(y_test,yhat))





























