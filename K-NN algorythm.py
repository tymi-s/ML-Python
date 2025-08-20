import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

pd.set_option('display.max_columns', None)
pd.set_option('display.width',250)
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')
print(df.head())

print(df['custcat'].value_counts())

#checking varius data corelation:

corelation_matrix = df.corr()

plt.figure()
sns.heatmap(corelation_matrix,annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5    )
plt.show()


#checking which values are the best :

corelation_values = abs(df.corr()['custcat'].drop('custcat')).sort_values(ascending=False)
print(corelation_values)


#sepaarating data
X= df.drop('custcat', axis=1)
y= df['custcat']

#normalising:

X_norm = StandardScaler().fit_transform(X)

#traintest split:

X_train,X_test,y_train,y_test = train_test_split(X_norm,y,test_size=0.2,random_state=4)


#CLASSYFICATION AND TRAINING:

for k in (1,2,3,4,5,6):
    #training:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_model = knn_classifier.fit(X_train,y_train)

    #predicting:

    yhat = knn_model.predict(X_test)

    #checking acuracy of a model using accuracy classyfication score:

    print(f"ACURACY SCORE WITH K = {k} ---", accuracy_score(y_test, yhat))

