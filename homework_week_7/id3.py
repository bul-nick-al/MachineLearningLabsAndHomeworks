import json

import numpy as np
import pandas as pd
import math
# code is based on this resource - http://gabrielelanaro.github.io/blog/2016/03/03/decision-trees.html
from sklearn.metrics import accuracy_score

def partition(a):
    return {c: (a == c).nonzero()[0] for c in np.unique(a)}


def partition_cont(a, pivot):
    """
    partitions a set of continuous data into two subsets
    :param a: a set a continuous data
    :param pivot: splitting point
    :return: a dictionary where the keys are two tuples, representing intervals: (-inf, pivot), (inf, pivot) and
    the corresponding subsets with indices in values.
    """
    return {(-np.inf, pivot): np.where(a < pivot)[0], (pivot, np.inf): np.where(a >= pivot)[0]}


def entropy(s):
    res = 0
    val, counts = np.unique(s, return_counts=True)
    freqs = counts.astype('float') / len(s)
    for p in freqs:
        if p != 0.0:
            res -= p * np.log2(p)
    return res


def information_gain(y, x):
    res = entropy(y)

    # We partition x, according to attribute values x_i
    val, counts = np.unique(x, return_counts=True)
    freqs = counts.astype('float') / len(x)

    # We calculate a weighted average of the entropy
    for p, v in zip(freqs, val):
        res -= p * entropy(y[x == v])

    return res


def information_gain_cont(y, x, pivot):
    """
    finds information gain for continuous data
    :param y:
    :param x:
    :param pivot:
    :return: information gain
    """

    # using the formula from the lab slides
    upper_entropy = entropy(y)
    prob1 = len(x[x < pivot])/len(x)
    prob2 = len(x[x >= pivot]) / len(x)
    branch_entropy = prob1 * entropy(y[x < pivot]) + prob2 * entropy(y[x >= pivot])

    return upper_entropy - branch_entropy


def information_gain_ratio(y, x, is_numeric):
    """
    Finds the information gain ration for a particular feature
    :param y:
    :param x: list of values of samples for the chosen feature
    :param is_numeric:
    :return: information gain ratio
    """
    if is_numeric:
        x = x.astype('float')
        unique_vals = np.unique(x)
        unique_vals = np.sort(unique_vals)
        split_candidates = []

        # make a list of possible pivots. Each pivot is the mean of two consequent values.
        for i in range(len(unique_vals)-1):
            split_candidates.append((unique_vals[i] + unique_vals[i+1])/2)
        ratio = 0
        best_split = None

        # find the max information gain ratio, splitting the set bby each pivot candidates
        for split_candidate in split_candidates:
            r_temp = information_gain_cont(y, x, split_candidate)/intrinsic_value_cont(x, split_candidate)
            if r_temp > ratio:
                ratio = r_temp
                best_split = split_candidate
        return ratio, best_split
    else:
        ratio = information_gain(y, x)/intrinsic_value(x)
        return (ratio if not math.isnan(ratio) else 0.0), None


def intrinsic_value(x):
    """
    Calculates the intrinsic value for x
    :param x: values of samples for a chosen feature
    :return: the intrinsic value
    """
    # We partition x, according to attribute values x_i
    val, counts = np.unique(x, return_counts=True)
    freqs = counts.astype('float') / len(x)

    res = 0

    # We calculate a weighted average of the entropy
    for p, v in zip(freqs, val):
        res -= p * np.log2(p)
    return res


def intrinsic_value_cont(x, pivot):
    """
    Calculates the intrinsic value for x
    :param pivot: the splitting value
    :param x: values of samples for a chosen feature
    :return: the intrinsic value
    """
    prob1 = len(x[x < pivot]) / len(x)
    prob2 = len(x[x >= pivot]) / len(x)

    return -(prob1 * np.log2(prob1) + prob2 * np.log2(prob2))


def is_pure(s):
    return len(set(s)) == 1


def recursive_split(x, y, fields, current_depth=0, max_depth=10):
    """
    :param x:
    :param y:
    :param fields: list with two lists: 1 - features' names, 2 - features' type, true - if numeric, false - categorical
    :param current_depth:
    :param max_depth:
    :return:
    """
    # If there could be no split, just return the original set
    if is_pure(y) or len(y) == 0:
        return y

    # We get attribute that gives the highest information gain
    gain = np.array([information_gain_ratio(y, x_attr, x_type) for x_attr, x_type in zip(x.T, fields[1])])

    selected_attr = np.argmax(gain[:, 0])
    # If there's no gain at all, nothing has to be done, just return the original set
    if np.all(gain[:, 0] < 1e-6):
        return y

    # tree pruning - we don't let the tree grow in depth more than the max depth
    if current_depth > max_depth:
        # print("max depth exceeded")
        return y

    # We split using the selected attribute
    if fields[1][selected_attr]:
        sets = partition_cont(x[:, selected_attr], gain[selected_attr][1])
    else:
        sets = partition(x[:, selected_attr])

    branches = {}
    for k, v in sets.items():
        y_subset = y.take(v, axis=0)
        x_subset = x.take(v, axis=0)

        branches[k] = recursive_split(x_subset, y_subset, fields, current_depth=current_depth+1, max_depth=max_depth)

    res = (fields[0][selected_attr], branches)
    return res


def predict_one_x(tree, x, fields):
    """
    Predicts outcome for a particular sample
    :param tree: a decision tree
    :param x: a sample
    :param fields: names of features and their types:
    :return:
    """

    # check if it is a leave node
    if type(tree) is np.ndarray:
        (values, counts) = np.unique(tree, return_counts=True)
        return values[np.argmax(counts)]

    feature_name = tree[0]  # get the name of the current feature
    is_numeric = fields[1][fields[0].index(feature_name)]  # get the type of the current feature
    if is_numeric:
        value = dict(zip(fields[0], x))[feature_name]  # get the value of the sample for that feature
        pivot = list(tree[1].keys())[0][1]  # get the split point
        if value < pivot:
            return predict_one_x(tree[1][(-np.inf, pivot)], x, fields)
        else:
            return predict_one_x(tree[1][(pivot, np.inf)], x, fields)
    else:
        branch = dict(zip(fields[0], x))[feature_name]  # get the class of the feature for this sample
        return predict_one_x(tree[1][branch], x, fields)


def predict(tree, x, fields):
    """
    makes predictions for samples
    :param tree: a decision tree
    :param x: an array of samples, whose outcomes are to be predicted
    :param fields: names of features and their types
    :return: a list with predictions for each sample
    """
    y = []
    for sample in x:
        y.append(predict_one_x(tree, sample, fields))
    return y


def preprocess_data(x, fields_types):
    """
    replaces empty cells with the mean of the feature in case of continuous feature and with the
    :param x:
    :param fields_types:
    :return:
    """
    for i, (column, is_numeric) in enumerate(zip(x.T, fields_types)):
        if is_numeric:
            column = column.astype('float')
            inds = np.where(np.isnan(column))
            mean = np.nanmean(column)
            x[:, i][inds] = mean
        else:
            column_temp = column.astype('str')
            inds = np.where(column_temp == 'nan')
            (values, counts) = np.unique(column_temp, return_counts=True)
            ind = np.argmax(counts)

            x[:, i][inds] = column[ind]
    return x


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

train_set = pd.read_csv('titanic_modified.csv')

fields = list(train_set.columns.values)
fields_types = [False, False, True, True, True, False]

X = preprocess_data(train_set.iloc[:, :6].values, fields_types)
y = train_set.iloc[:, 6].values

# I've picked the max depth = 21, because this is the smallest depth, which gives the accuracy > 90%
tree = recursive_split(X, y, [fields, fields_types], max_depth=21)
prediction = predict(tree, X, [fields, fields_types])


print("\n---------------------------------------\nThe tree:\n{}".format(tree))
print("\n---------------------------------------\nThe predictions:\n{}".format(prediction))
print("\n---------------------------------------\nThe accuracy of prediction is {}"
      .format(accuracy_score(y, prediction)))



