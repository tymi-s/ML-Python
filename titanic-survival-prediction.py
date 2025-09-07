"""
INTRODUCTION

Now that you have a feel for how to optimize your machine learning pipeline, let's practice with a real world dataset.
You'll use cross validation and a hyperparameter grid search to optimize your machine learning pipeline.

You will use the Titanic Survival Dataset to build a classification model to predict whether
a passenger survived the sinking of the Titanic, based on attributes of each passenger in the data set.

You'll start with building a Random Forest Classifier, then modify your pipeline to use a Logistic Regression estimator instead.
You'll evaluate and compare your results.



"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

titanic = sns.load_dataset("titanic")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)
print(titanic.head())


print(titanic.count())


#we droping DECK becouse it has a lot of null values as well as AGE.  Embarked and embark_town seams not to be relevant
#marking which data interest us :
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'class', 'who', 'adult_male', 'alone']

#survived is a target var
target  ='survived'

X=titanic[features]
y=titanic[target]

#Checking how balanced are the classes:
print(y.value_counts())#549 passengers did not survive, 342 has survived
#38% of passengers survived. Data are a bit imbalanced so we will have to stratify them during train-test split and cross-validation

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)#train test split with stratifying


"""
Define preprocessing transformers for numerical and categorical features
Automatically detect numerical and categorical columns and assign them to separate numeric and categorical features

"""

numerical_features=X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features=X_train.select_dtypes(exclude=['object','category']).columns.tolist()


"""
defining preprocessing pipelines for both variable types:
"""


numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

"""
Combine the transformers into a single column transformer
"""

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])


"""
creating a model pipeline
Now let's complete the model pipeline by combining the preprocessing with a Random Forest classifier
"""

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])


"""
Define a parameter grid
"""

param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}



"""
Perform grid search cross-validation and fit the best model to the training data
"""

cv = StratifiedKFold(n_splits=5, shuffle=True)


"""
Train the pipeline model
"""


model = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=2)
model.fit(X_train, y_train)

"""
 Get the model predictions from the grid search estimator on the unseen data
"""

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


"""
Plot the confusion matrix
"""
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')

# Set the title and labels
plt.title('Titanic Classification Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')


plt.tight_layout()
plt.show()

"""
FEATURE IMPORTANCE

Let's figure out how to get the feature importances of our overall model.
First, to obtain the categorical feature importances,
we have to work our way backward through the modelling pipeline to associate the feature importances 
with their one-hot encoded input features that were transformed from the original categorical features.

We don't need to trace back through the pipeline for the numerical features, because we didn't transfrom them into new ones in any way.
Remember, we went from categorical features to one-hot encoded features, using the 'cat' column transformer.

"""
# here is a process of tracing back hrough the trained model to access the one-hot encoded feature names:
model.best_estimator_['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)


"""
Notice how the one-hot encoded features are named - for example, sex was split into two boolean features indicating whether the sex is male or female.

Great! Now let's get all of the feature importances and associate them with their transformed feature names.
"""

feature_importances = model.best_estimator_['classifier'].feature_importances_

# Combine the numerical and one-hot encoded categorical feature names
feature_names = numerical_features + list(model.best_estimator_['preprocessor']
                                        .named_transformers_['cat']
                                        .named_steps['onehot']
                                        .get_feature_names_out(categorical_features))

"""
Display the feature importances in a bar plot
Define a feature importance DataFrame, then plot it
"""
importance_df = pd.DataFrame({'Feature': feature_names,
                              'Importance': feature_importances
                             }).sort_values(by='Importance', ascending=False)

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.gca().invert_yaxis()
plt.title('Most Important Features in predicting whether a passenger survived')
plt.xlabel('Importance Score')
plt.show()

# Print test score
test_score = model.score(X_test, y_test)
print(f"\nTest set accuracy: {test_score:.2%}")




