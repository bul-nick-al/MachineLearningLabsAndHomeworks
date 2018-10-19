import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np



def read_and_clean_data():
    df = pd.read_csv('wine.csv')
    return df



def calc_ridge(X, y, alphas, plot):
    kf = KFold(n_splits=10)
    kf.get_n_splits(X)
    mses = []
    for alpha in alphas:
        errors = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            ridge = Ridge(normalize=True)
            ridge.alpha = alpha
            ridge.fit(X_train, y_train)
            pred_y = ridge.predict(X_test)
            errors.append(mean_squared_error(y_true=y_test, y_pred=pred_y))
        mses.append(np.mean(errors))
    if plot:
        plt.plot(alphas, mses, 'ro')
        plt.title("MSE for different alpha levels for Ridge Regression")
        plt.xlabel('alpha')
        plt.ylabel('MSE')
        plt.xscale('log')
        plt.show()
    return mses


def calc_lasso(X, y, alphas, plot):
    kf = KFold(n_splits=10)
    kf.get_n_splits(X)
    mses = []
    for alpha in alphas:
        errors = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            lasso = Lasso(normalize=True)
            lasso.alpha = alpha
            lasso.fit(X_train, y_train)
            pred_y = lasso.predict(X_test)
            errors.append(mean_squared_error(y_true=y_test, y_pred=pred_y))
        mses.append(np.mean(errors))
    if plot:
        plt.plot(alphas, mses, 'ro')
        plt.title("MSE for different alpha levels for Lasso Regression")
        plt.xlabel('alpha')
        plt.ylabel('MSE')
        plt.xscale('log')
        plt.show()
    return mses



def knn(X,y, plot):
    kf = KFold(n_splits=10)
    kf.get_n_splits(X)
    mses = []

    for k in range(1, 40):
        errors = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            pred_y = knn.predict(X_test)
            errors.append(mean_squared_error(y_true=y_test, y_pred=pred_y)/len(y_test))
        mses.append(np.mean(errors))
    if plot:
        plt.plot(range(1, 40), mses, 'ro')
        plt.title("MSE for different K levels KNN")
        plt.xlabel('k')
        plt.ylabel('MSE')
        # plt.xscale('log')
        plt.show()
    return mses


def compare(X, y, ringe_alpha, lasso_alpha, k, plot):
    kf = KFold(n_splits=10)
    kf.get_n_splits(X)
    knn_errors = []
    ridge_errors = []
    lasso_errors = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        pred_y = knn.predict(X_test)
        knn_errors.append(mean_squared_error(y_true=y_test, y_pred=pred_y))

        lasso = Lasso(normalize=True)
        lasso.alpha = lasso_alpha
        lasso.fit(X_train, y_train)
        pred_y = lasso.predict(X_test)
        lasso_errors.append(mean_squared_error(y_true=y_test, y_pred=pred_y))

        ridge = Ridge(normalize=True)
        ridge.alpha = ringe_alpha
        ridge.fit(X_train, y_train)
        pred_y = ridge.predict(X_test)
        ridge_errors.append(mean_squared_error(y_true=y_test, y_pred=pred_y))

    if plot:
        plt.plot([0, 1, 2], [np.mean(knn_errors), np.mean(ridge_errors), np.mean(lasso_errors)], 'ro')
        plt.title("Comparison")
        plt.xlabel('models (knn - 0, ridge - 2, lasso - 3)')
        plt.ylabel('MSE')
        # plt.xscale('log')
        plt.show()
    return np.mean(knn_errors), np.mean(ridge_errors), np.mean(lasso_errors)

df = read_and_clean_data()
y = df.quality
X = df.drop('quality', axis=1)
knn(X, y, True)
alphas = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]
calc_ridge(X, y, alphas, True)
calc_lasso(X, y, alphas, True)
compare(X, y, 1e-2, 1e-4, 20, True)\


def ani(data_x, data_y, true_b1, true_b0, b1, b0, x_range=(-10, 10), label = ''):
    plt.scatter(data_x, data_y)
    plt.plot([x_range[0], x_range[1]],
             [x_range[0]*true_b1 + true_b0, x_range[1]*true_b1 + true_b0], c='r', linewidth=2, lable='True')
    plt.plot







