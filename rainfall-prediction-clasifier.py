import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns


url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
df = pd.read_csv(url)
df.head()
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 350)
print(df.count())

#sunshine i cloud cover wydają się być istotne ale mają dużo braków danych i jest ich zbyt dużo żeby je przypisać
# teraz będzie usuwanie wszystkich rzeędów w któych występują missing values:
df = df.dropna()
print(df.info())

#po usunięciu missing values mamy 56420 pomiarów. Sprawdzenie jak to wyglada:
print("\n\n\n")
print(df.columns)

#zmiana nazw żeby zapobiec kłopotom rozumowania:

df = df.rename(columns={'RainToday': 'RainYesterday',
                        'RainTomorrow': 'RainToday'
                        })


#ciężko przewidzieć deszcz w całej australii bo jest to duży obszar wymagałoby to dużo pamięci i danych. Trzeba sprawdzić ile obserwacji mamy na dany
#fragment Australii i wybrać ten w którym jest najwięcej:

df = df[df.Location.isin(['Melbourne','MelbourneAirport','Watsonia'])];
print(df.info())

# w tym obszarze mamy 7557 obserwacji i jest to wystarczająco żeby zbudować dobry model;

#dobra to czy będzie padać zależy od pory roku np na jesien pada więcej niż we wiosne, trzeba rozdzielić dane zależnie od pór roku:
def date_to_season(date):
    month = date.month
    if (month == 12) or (month == 1) or (month == 2):
        return 'Summer'
    elif (month == 3) or (month == 4) or (month == 5):
        return 'Autumn'
    elif (month == 6) or (month == 7) or (month == 8):
        return 'Winter'
    elif (month == 9) or (month == 10) or (month == 11):
        return 'Spring'

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Apply the function to the 'Date' column
df['Season'] = df['Date'].apply(date_to_season)
df.drop(columns=['Date'],inplace=True)
print(df)




