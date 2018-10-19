import sklearn

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation, rc


class LosgisticRegression:
    def __init__(self, x, y, a=1, b=0, lr=0.01, iterations=1000):
        self.x = x
        self.y = y
        self.a = a
        self.b = b
        self.lr = lr
        self.iterations = iterations
        # First set up the figure, the axis, and the plot element we want to animate
        self.fig, self.ax = plt.subplots()

        self.ax.set_xlim((0, 2))
        self.ax.set_ylim((-2, 2))

        self.line, = self.ax.plot([], [], lw=2)
        self.ax.scatter([0.1, 0.2, 1.7, 1.8], [0, 0, 1, 1])

        # call the animator. blit=True means only re-draw the parts that have changed.


    # initialization function: plot the background of each frame
    def init(self):
        self.line.set_data([], [])
        return (self.line,)

    def fit(self):
        for i in range(self.iterations):
            # if i % 1 == 0:
                # print("           ", self.accuracy(self.y, self.predict(self.x)))

            self.a = self.a - self.lr * self.gradient_descent_a(self.a)
            self.b = self.b - self.lr * self.gradient_descent_b(self.b)

    def gradient_descent_a(self, a):
        return (self.x * (self.p(self.x) - self.y)).mean()

    def gradient_descent_b(self, b):
        return (self.p(self.x) - self.y).mean()

    def p(self, x):
        z = np.exp(self.a * x + self.b) / (1 + np.exp(self.a * x + self.b))
        return z

    def getClass(self, x):
        return 1 if self.p(x) > 0.5 else 0

    def accuracy(self, Y, y):
        val = 0
        for i, k in zip(Y, y):
            if (i == 1 and k) or (i == 0 and not k):
                val += 1
        return val / len(Y)

    def predict(self, x):
        return self.p(x) > 0.5


def gradient_descent(X, y):
    b1, b0 = 1, 0
    alpha = 0.001
    prev_step = 1
    precision = 0.1
    while prev_step > precision:
        b0_prev, b1_prev = b0, b1
        b0 = b0 - alpha * part_der_b0(b0, b1, X, y)
        b1 = b1 - alpha * part_der_b1(b0, b1, X, y)
        prev_step = abs(b0 - b0_prev) + abs(b1 - b0_prev)
    return b0, b1


def part_der_b0(b0, b1, X, Y):
    return (sigma(b0, b1, X) - Y).mean()


def part_der_b1(b0, b1, X, Y):
    return (X*(sigma(b0, b1, X) - Y)).mean()


def sigma(b0, b1, x):
    return np.exp(b0 + b1 * x) / (1 + np.exp(b0 + b1 * x))


iris = load_iris()
print(iris)

# iris = pd.read_csv("/Users/nicholas/Downloads/iris.csv")
#
# iris = iris.drop(['petal_width', 'sepal_width', 'petal_length'], axis=1)
# iris = iris[iris.species != 'setosa']
# iris = iris.replace({'versicolor': 0, 'virginica': 1})
#
# iris['species'] = pd.to_numeric(iris['species'])
# iris['sepal_length'] = pd.to_numeric(iris['sepal_length'])
# y = iris.species
# X = iris.drop(['species'], axis=1)
#
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# lr = LosgisticRegression(np.array(X_test), np.array(y_test))
# lr.fit()
# print(lr.a, lr.b)
# b1, b0 = gradient_descent(np.array(X_test), np.array(y_test))
# lrr = LogisticRegression()
# lrr.fit(np.array(X_test), np.array(y_test))
# print(lrr.intercept_, lrr.coef_)
# print(b0, b1)

