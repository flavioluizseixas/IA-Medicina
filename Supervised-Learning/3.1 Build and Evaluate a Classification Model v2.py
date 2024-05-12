import os

# Core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot

# Sklearn functionality
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Convenience functions.  This can be found on the course github
from functions import *

# Load the data set
dataset = pd.read_csv("./data/csv/world_data.csv")

# Examine the date shape
dataset.shape
# Inspect first few rows
dataset.head()
# Check data types
dataset.dtypes

# Check for nulls
dataset.isnull().mean().sort_values(ascending=False)
# Discard very sparse features
dataset = dataset.drop(["murder","urbanpopulation","unemployment"], axis=1)

# Função para imputar a mediana nas colunas float
def impute_median_float(df):
    for col in df.select_dtypes(include='float'):
        median = df[col].median()
        df.fillna({col: median}, inplace=True)

impute_median_float(dataset)

# Bin lifexp into L, M and H bands
dataset = appendEqualCountsClass(dataset, "lifeexp_band", "lifeexp", 3, ["L","M","H"])
# Check how many rows in each bin
dataset.lifeexp_band.value_counts()

# Split into input and target features
y = dataset["lifeexp_band"]
X = dataset[['happiness', 'income', 'sanitation', 'water', 'literacy', 'inequality', 'energy', 'childmortality', 'fertility',  'hiv', 'foodsupply', 'population']]

X.describe()
# Rescale the data
scaler = MinMaxScaler(feature_range=(0,1))
rescaledX = scaler.fit_transform(X)

# Convert X back to a Pandas DataFrame, for convenience
X = pd.DataFrame(rescaledX, columns=X.columns)
X.describe()

# Split into test and training sets
test_size = 0.33
seed = 1
X_train, X_test, Y_train, Y_test =  train_test_split(X, y, test_size = test_size, random_state = seed)

# Build a decision tree model
model = DecisionTreeClassifier()

# Definindo o grid de hiperparâmetros para serem testados
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Criando o objeto GridSearchCV
grid_search = GridSearchCV(estimator = model, param_grid = param_grid, cv = 5)

# Treinando o GridSearchCV
grid_search.fit(X, y)

# Imprimindo os melhores hiperparâmetros encontrados
print("Melhores hiperparâmetros encontrados:")
print(grid_search.best_params_)

# Acessando o melhor modelo encontrado
best_model = grid_search.best_estimator_

predictions = best_model.predict(X_test)
print("DecisionTreeClassifier", accuracy_score(Y_test, predictions))

viewDecisionTree(best_model, X.columns)

#scores = cross_val_score(model_dt, X, y, cv = 5)
#model_dt.fit(X_train, Y_train)

#model_dt.fit(X_train, Y_train)
# Check the model performance with the training data
#predictions_dt = model_dt.predict(X_train)
#print("DecisionTreeClassifier", accuracy_score(Y_train, predictions_dt))

#predictions_dt = model_dt.predict(X_test)
#print("DecisionTreeClassifier", accuracy_score(Y_test, predictions_dt))

#viewDecisionTree(model_dt, X.columns)
