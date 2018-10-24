import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import Imputer, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def prepare_data(df):
    """
    Peprapes data for further processing
    :param df: data
    :return: X and y, where Regions and countries are removed from X and y only has 1 - for Europe and 0 otherwise
    """
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    X = df.drop(['Region', 'Country'], axis=1)  # get rid of correlated and dependent features
    X = imp.fit_transform(X)  # handle missing values

    y = df["Region"]
    warnings.simplefilter("ignore")  # the following line is correct but may cause warnings
    y[y != "EUROPE"] = "OTHERS"  # replace all regions but Europe with OTHERS
    y = y.replace({"EUROPE": 1, "OTHERS": 0}).astype('int64')
    return np.array(X), np.array(y)


def perform_pca(X, n_components):
    """
    Performs PCA and returns 14 first components
    :param X: the original, to some extend correlated data
    :return: first 14 principal components
    """
    pca = PCA(n_components=n_components)
    pca.fit(X)
    return pca.transform(X)


def calculate_w(error):
    """
    Calculates weight of some particular estimator in AdaBoost, given the weighted error for this estimator
    :param error: the weighted error for the estimator
    :return: weight w
    """
    return 0.5 * np.log((1 - error)/error)  # based on the formula from the slides, Lecture 8


def calculate_sample_weights(model, X, y):
    """
    Calculates the weights (alphas) for each sample in the set after the AdaBoost model is trained on it
    :param model: the trained AdaBoost model
    :param X: training data set
    :param y:
    :return: an array of weights
    """
    sample_weights = np.array([1/len(X) for _ in range(len(X))])  # test the initial weights to 1/#_samples

    # iterate over estimators in the model and the errors for those estimators
    for error, estimator in zip(model.estimator_errors_, model.estimators_):
        w = calculate_w(error)
        y_predicted = estimator.predict(X)
        misclassified_indices = np.where(y_predicted != y)[0]  # compare predicted values and the actual values

        # recalculate weights
        for i in range(len(sample_weights)):
            if i in misclassified_indices:
                sample_weights[i] = sample_weights[i] * np.exp(w)  # for the misclassified samples
            else:
                sample_weights[i] = sample_weights[i] * np.exp(-w)  # for the correctly classified samples
        sample_weights /= sum(sample_weights)

    return sample_weights


def calculate_bound(weights):
    return np.mean(weights) + 2*np.std(weights)


def find_inliers_using_adaboost(X, y):
    """
    performs AdaBoost on X and based on the result of training, gets rid of the outliers
    :param X:
    :param y:
    :return: indices of inliers
    """

    # training AdaBoost
    classifier = AdaBoostClassifier(base_estimator=LogisticRegression(), n_estimators=20)
    classifier.fit(X, y)
    weights = calculate_sample_weights(classifier, X, y)

    # calculate the line, above which all point will be count as outliers
    bound = calculate_bound(weights)

    # plot the coefficients and the decision line
    plt.scatter([i for i in range(len(weights))], weights)
    plt.plot([i for i in range(len(weights))], [bound for _ in range(len(weights))], color='r')
    plt.ylabel('Coefficients')
    plt.xlabel('Samples')
    plt.title("Coefficients of the samples and the decision line")
    plt.show()

    return np.where(weights < bound)[0]  # find the inliers' indices


def cv_adaboost_with_and_without_outliers(X, y, inliers_indices):
    """
    Performs cross validation on adaboost for sets with and without outliers in order to estimate difference in their
    accuracy
    :param X:
    :param y:
    :param inliers_indices:
    :return:
    """
    classifier = AdaBoostClassifier(base_estimator=LogisticRegression(), n_estimators=20)
    # accuracies will be saved here
    accuracy_with_outliers = []
    accuracy_without_outliers = []

    kf = KFold(n_splits=20)

    # perform cv on the set with the outliers
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier.fit(X_train, y_train)
        prediction = classifier.predict(X_test)
        accuracy_with_outliers.append(accuracy_score(y_test, prediction))

    # perform cv on the set without the outliers
    X, y = X[inliers_indices], y[inliers_indices]
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier.fit(X_train, y_train)
        prediction = classifier.predict(X_test)
        accuracy_without_outliers.append(accuracy_score(y_test, prediction))

    print("Avg accuracy with outliers: {0:0.3f}".format(np.mean(accuracy_with_outliers)))
    print("Avg accuracy without outliers: : {0:0.3f}".format(np.mean(accuracy_without_outliers)))

    # return difference between
    return np.mean(accuracy_without_outliers) - np.mean(accuracy_with_outliers)


def plot_components_correlation(X, title):
    """
    Prints the corr matrix (this code was taken from the Internet)
    :param X: data
    :param title:
    """
    columns = [str(x) for x in np.arange(1, X.shape[1], 1)]
    data = pd.DataFrame({key: values for key, values in zip(columns, X.T)})
    plt.subplots(figsize=(20, 20))
    plt.title(title)
    sns.heatmap(data.astype(float).corr(), linewidths=0.25, vmax=1.0, square=True,
                cmap="YlGnBu", linecolor='black', annot=True)
    plt.show()


def plot_explained_variance(X):
    cov_mat = np.cov(X.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    # Calculation of Explained Variance from the eigenvalues
    tot = sum(eig_vals)
    var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance
    cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(var_exp)), var_exp, label='explained variance of each PC', color = 'r')
    plt.step(range(len(cum_var_exp)), cum_var_exp, where='mid', label='cumulative explained variance')
    plt.ylabel('Explained variance')
    plt.xlabel('Principal components')
    plt.title("Cumulative and individual explained variance for personal components")
    plt.show()


def scale_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)


countries_data = pd.read_csv('countries.csv')
X, y = prepare_data(countries_data)
X = scale_data(X)

plot_components_correlation(X, "Correlation of the features")
plot_explained_variance(X)

X = perform_pca(X, n_components=12)

plot_components_correlation(X, "Correlation of the Principal Components")

inliers_indices = find_inliers_using_adaboost(X, y)
print("The difference of accuracies: {0:0.3f}".format(cv_adaboost_with_and_without_outliers(X, y, inliers_indices)))
