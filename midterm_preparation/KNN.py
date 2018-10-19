from math import sqrt

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier




class KNN:
    def __init__(self, n_neighbours=1):
        self.k = n_neighbours
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        predictions = []
        # distances = np.zeros(shape=X.shape[0])
        for row in X:
            predictions.append(self.predict_one(row))
        return predictions

    def predict_one(self, x):
        distances = []
        for row in self.X:
            distances.append(self.calc_distance(row, x))

        sorted_distances = np.sort(distances)
        sorted_distances = sorted_distances[0:self.k]

        classes = []
        for sorted_distance in sorted_distances:
            index = distances.index(sorted_distance)
            classes.append(self.y[index])

        (values, counts) = np.unique(np.array(classes), return_counts=True)
        ind = np.argmax(counts)
        return values[ind]

    def calc_distance(self, x1, x2):
        sum_of_squares = 0
        for x1_f, x2_f in zip(x1, x2):
            sum_of_squares += (x1_f - x2_f) ** 2
        return sqrt(sum_of_squares)


def read_and_clean_data():
    df = pd.read_csv('wine.csv')
    return df

df = read_and_clean_data()
y = df.quality
X = df.drop('quality', axis=1)
knn = KNN(n_neighbours=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
knn.fit(np.array(X), np.array(y))
c = np.array(knn.predict(np.array(X)))
v = np.where(c == y)
print(len(np.where(c == y)[0])/len(y))

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(np.array(X), np.array(y))
c = np.array(knn.predict(np.array(X)))
print(len(np.where(c == y)[0])/len(y))

