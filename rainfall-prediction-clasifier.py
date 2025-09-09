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


X = df.drop(columns=['RainToday'],axis=1)
y = df['RainToday']
# spraawdzanie jak zbalanowane są dane:
print(y.value_counts())


"""
WNIOSKI Z BALANSU DANYCH:

## Write your response here and convert the cell to a markdown.
Based on the dataset, rain occurs on 1,791 out of 7,557 days, which is about 23.7% of the time.
This means that it rains on roughly one out of every four days in the Melbourne area.

If we simply predicted "No rain" for every day, we would be correct 5,766 out of 7,557 times,
which gives an accuracy of about 76.3%.
However, this model would completely fail to detect rainy days (0% recall for the "Yes" class).

No, this is an imbalanced dataset because the "No" class is much more frequent than the "Yes" class
(about 4 times more common).
Class imbalance can lead to biased models that predict "No" too often.

Resampling methods – oversample the "Yes" cases (SMOTE) or undersample "No".
Class weights – use models that support class_weight='balanced'
Train a baseline model and evaluate it using precision, recall, F1-score,
not just accuracy, since accuracy can be misleading on imbalanced data.                                      
Perform feature engineering (e.g., seasonality, weather conditions) to improve predictive power  

"""

#train-test split with stratyfication:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#preprocessing transformers with auto identifying a numerical and categorical features
#1 - Automatically detect numerical and categorical columns and assign them to separate numeric and categorical features

numeric_features = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

#transormacje dla każdego z typów danych - Define separate transformers for both feature types and combine them into a single
# preprocessing transformer:

# Scale the numeric features
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

# One-hot encode the categoricals
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])



#Combine the transformers into a single preprocessing column transformer


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)


#PIPELINE:
# Create a pipeline by combining the preprocessing with a Random Forest classifie:

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

#Define a parameter grid to use in a cross validation grid search model optimizer

param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

"""
Pipeline usage in crossvalidation:

Recall that the pipeline is repeatedly used within the crossvalidation
by fitting on each internal training fold and predicting on its corresponding validation fold

"""

#Perform grid search cross-validation and fit the best model to the training data
#1. Select a cross-validation method, ensuring target stratification during validation

cv = StratifiedKFold(n_splits=5, shuffle=True)
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)


#Print the best parameters and best crossvalidation score
print("\nBest parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


#Display your model's estimated score
test_score = grid_search.score(X_test, y_test)
print("Test set score: {:.2f}".format(test_score))

#Get the model predictions from the grid search estimator on the unseen data
y_pred = grid_search.predict(X_test)


#Print the classification report
print("\nClassification Report:")
print(classification_report (y_test, y_pred))

#ploting a confusion matrix:

conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

#wyniki są słabe:
# 1 - będzie padać przewidziano 183 poprawnie i 175 nie poprawnie
# 2 - nie będzie padać przewidziano 1094 poprawnie i 60 niepoprawnie

#poprawki:
#feature importance - trzeba się teraz dużo conąć w pipelinie żeby to wziąść
# pod uwagę
"""

Feature importances¶
Recall that to obtain the categorical feature importances, we have to work our way backward through the modelling
pipeline to associate the feature importances with their original input variables, not the one-hot encoded ones.
We don't need to do this for the numeric variables because we didn't modify their names in any way.
Remember we went from categorical features to one-hot encoded features, using the 'cat' column transformer.

Let's get all of the feature importances and associate them with their transformed features

"""


#Extract the feature importances
feature_importances = grid_search.best_estimator_['classifier'].feature_importances_


#Now let's extract the feature importances and plot them as a bar graph.
# Combine numeric and categorical feature names
feature_names = numeric_features + list(grid_search.best_estimator_['preprocessor']
                                        .named_transformers_['cat']
                                        .named_steps['onehot']
                                        .get_feature_names_out(categorical_features))

feature_importances = grid_search.best_estimator_['classifier'].feature_importances_

importance_df = pd.DataFrame({'Feature': feature_names,
                              'Importance': feature_importances
                             }).sort_values(by='Importance', ascending=False)

N = 20  # Change this number to display more or fewer features
top_features = importance_df.head(N)

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature on top
plt.title(f'Top {N} Most Important Features in predicting whether it will rain today')
plt.xlabel('Importance Score')
plt.show()

#wnioski z feature importance:
#najważniejsze cechy to :
# humidity 3pm
# pressure 9am
# sunsine
#pressure 3pm


#próba z innym modelem:
# update pipelina z nowym modelem:
# tym razem to będzie logisitc regression:
# Replace RandomForestClassifier with LogisticRegression
pipeline.set_params(classifier=LogisticRegression(random_state=42))

# update the model's estimator to use the new pipeline
grid_search.estimator = pipeline

# Define a new grid with Logistic Regression parameters
param_grid = {
    # 'classifier__n_estimators': [50, 100],
    # 'classifier__max_depth': [None, 10, 20],
    # 'classifier__min_samples_split': [2, 5],
    'classifier__solver' : ['liblinear'],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight' : [None, 'balanced']
}

grid_search.param_grid = param_grid

# Fit the updated pipeline with LogisticRegression
grid_search.fit(X_train, y_train)

# Make predictions
y_pred = grid_search.predict(X_test)


#porównanie wyników z poprzednim modelem:

print(classification_report(y_test, y_pred))

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')

# Set the title and labels
plt.title('Titanic Classification Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show the plot
plt.tight_layout()
plt.show()




"""
Comperition:
Acuracy:
- RandomForestClassifier: 84%
- LogisticRegression: 83%

Number of correct predictions:
- RandomForestClassifier: 1337
- LogisticRegression: 1337

Recall:
- RandomForestClassifier: 51
- LogisticRegression: 51
"""