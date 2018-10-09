import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from lab1.exploring_libraries import split_to_sets

df = pd.read_csv('../mpg.csv')



df = df.drop('name', axis=1)
df['origin'] = df['origin'].replace({1: 'america', 2: 'europe', 3: 'asia'})
df = pd.get_dummies(df, columns=['origin'])
df = df.replace('?', np.nan)
df = df.dropna()



(training_set, testing_set) = split_to_sets(df, 0.25)

X_train = training_set.drop('mpg', axis=1)
X_test = testing_set.drop('mpg', axis=1)
y_train = training_set[['mpg']]
y_test = testing_set[['mpg']]

regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

print(regression_model.score(X_test, y_test))

# with pd.option_context('display.max_rows', 100, 'display.max_columns', 20, 'display.width', 1000):
#     print(df.head(100))