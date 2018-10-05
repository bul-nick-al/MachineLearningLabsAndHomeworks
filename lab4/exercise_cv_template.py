import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import random


# performs k-fold cross-validation of Ridge regression to find optimal value of alpha
# inputs: k - #folds, X, y - dataset, aplhas - list of alphas to choose from
# output: list of mse, for each of possible alphas
def ridge_cv(k, X, y, alphas):
    splits = k_split(k, X.shape[0])
    results = []
    for alpha in alphas:
        temp_ridges = []
        for i, split in enumerate(splits):
            X_train = X.iloc[split[1]]
            y_train = y.iloc[split[1]]
            X_test = X.iloc[split[0]]
            y_test = y.iloc[split[0]]
            ridge = Ridge(normalize=True)
            ridge.set_params(alpha=alpha)
            ridge.fit(X_train, y_train)
            predicted_y = ridge.predict(X_test)
            temp_ridges.append(mean_squared_error(y_true=y_test, y_pred=predicted_y))
        results.append(sum(temp_ridges)/len(temp_ridges))
    return results



def ridge(actual_y, predicted_y, coefs, alpha):
    betas = coefs
    betas = list(map(lambda x: x ** 2, betas))
    betas_sum = sum(betas)
    rss = mean_squared_error(y_true=actual_y, y_pred=predicted_y)
    return rss + alpha*betas_sum

# sub-procedure for ridge_cv
# returns k splits as tuples (train_indices, test_indices)
# inputs: k - #folds, l - #rows in the dataset (length)
# output: list of tuples
def k_split(k, l):
    splits = []
    indices = [i for i in range(l)]
    random.shuffle(indices)
    chunks = np.array_split(indices, k)
    for i, chunk in enumerate(chunks):
        train_set = chunks.copy()
        del train_set[i]
        train_set = [y for x in train_set for y in x]
        splits.append((chunk, train_set))

    return splits


# load and pre-process the dataset
hitters = pd.read_csv("Hitters.csv").dropna().drop("Player", axis=1)
dummies = pd.get_dummies(hitters[['League', 'Division', 'NewLeague']])
# Drop the column with the independent variable (Salary), and columns for which we created dummy variables
X_ = hitters.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')
# Define the feature set X.
X = pd.concat([X_, dummies[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)
y = hitters.Salary
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)
alphas = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]

results = ridge_cv(10, X, y, alphas)

# visualize the results
plt.plot(alphas, results, 'ro')
plt.title("MSE for different alpha levels for Ridge Regression")
plt.xlabel('alpha')
plt.ylabel('MSE')
plt.xscale('log')
plt.show()
