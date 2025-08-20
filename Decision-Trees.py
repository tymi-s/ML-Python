import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics



import warnings
warnings.filterwarnings('ignore')

###########################################################################################################################
# pobranie danych
path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
my_data = pd.read_csv(path)
pd.set_option('display.width', 250)
print(my_data.head())
print(my_data.info())

#dane kategoryczne (object) muszą być zmienione na numeryczne, aby działało modelowanie
#preprocessing:
label_encoder = LabelEncoder()
my_data['Sex'] = label_encoder.fit_transform(my_data['Sex'])
my_data['BP'] = label_encoder.fit_transform(my_data['BP'])
my_data['Cholesterol'] = label_encoder.fit_transform(my_data['Cholesterol'])



#zamiana na dane numeryczne ostatniej kolumny:
custom_map = {'drugA':0,'drugB':1,'drugC':2,'drugX':3,'drugY':4}
my_data['Drug_num'] = my_data['Drug'].map(custom_map)

columns = ['Age','Sex','BP','Cholesterol','Na_to_K','Drug_num']
print(my_data[columns].corr())



















