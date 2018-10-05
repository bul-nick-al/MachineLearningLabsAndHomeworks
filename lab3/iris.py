from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from lab3.Linear_regression_gradient_descent import gradient_descent

iris = pd.read_csv("/Users/nicholas/Downloads/iris.csv")

iris = iris.drop(['petal_width', 'sepal_width', 'petal_length'], axis=1)
iris = iris[iris.species != 'setosa']
iris = iris.replace({'versicolor': 0, 'virginica': 1})
y = iris.species
X = iris.drop(['species'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


b1, b0 = gradient_descent(X_train.sepal_length.values, y_train.values)
print()
# for x, y in zip(X_test, y_test):