import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

url =  "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
df = pd.read_csv(url)
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', None)
print(df.head())

LR = LogisticRegression()

df['churn'] = df['churn'].astype('int')

y=np.asarray(df['churn'])
X=np.asarray(df[['tenure', 'age', 'address', 'ed', 'equip']])

X_norm = StandardScaler().fit(X).transform(X)
X_train,X_test,y_train,y_test=train_test_split(X_norm,y,test_size=0.2,random_state=4)
LR.fit(X_train,y_train)


prediction= LR.predict_proba(X_test)

print(log_loss(y_test,prediction))
