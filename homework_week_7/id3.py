import json

import numpy as np
import pandas as pd
import math
# code is based on this resource - http://gabrielelanaro.github.io/blog/2016/03/03/decision-trees.html


def partition(a):
    return {c: (a == c).nonzero()[0] for c in np.unique(a)}
def partition_cont(a, pivot):
    return {(-np.inf, pivot): np.where(a < pivot)[0], (pivot, np.inf): np.where(a >= pivot)[0]}


def entropy(s):
    res = 0
    val, counts = np.unique(s, return_counts=True)
    freqs = counts.astype('float') / len(s)
    for p in freqs:
        if p != 0.0:
            res -= p * np.log2(p)
    return res


def entropy_cont(s):
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
    upper_entropy = entropy(y)
    prob1 = len(x[x < pivot])/len(x)
    prob2 = len(x[x >= pivot]) / len(x)
    branch_entropy = prob1*entropy(y[x < pivot]) + prob2*entropy(y[x >= pivot])

    return upper_entropy - branch_entropy


def information_gain_ratio(y, x, is_numeric):
    if is_numeric:
        x = x.astype('float')
        unique_vals = np.unique(x)
        unique_vals = np.sort(unique_vals)
        split_candidates = []
        for i in range(len(unique_vals)-1):
            split_candidates.append((unique_vals[i] + unique_vals[i+1])/2)
        ratio = 0
        best_split = None
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
    # We partition x, according to attribute values x_i
    val, counts = np.unique(x, return_counts=True)
    freqs = counts.astype('float') / len(x)

    res = 0

    # We calculate a weighted average of the entropy
    for p, v in zip(freqs, val):
        res -= p * np.log2(p)
    return res


def intrinsic_value_cont(x, pivot):
    prob1 = len(x[x < pivot]) / len(x)
    prob2 = len(x[x >= pivot]) / len(x)

    return -(prob1 * np.log2(prob1) + prob2 * np.log2(prob2))



def is_pure(s):
    return len(set(s)) == 1


def recursive_split(x, y, fields):
    # If there could be no split, just return the original set
    if is_pure(y) or len(y) == 0:
        return y

    # We get attribute that gives the highest information gain
    gain = np.array([information_gain_ratio(y, x_attr, x_type) for x_attr, x_type in zip(x.T, fields[1])])
    c = gain[:, 0]

    selected_attr = np.argmax(gain[:, 0])

    # If there's no gain at all, nothing has to be done, just return the original set
    if np.all(gain[:, 0] < 1e-6):
        print("hi")
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

        branches[k] = recursive_split(x_subset, y_subset, fields)

    res = (fields[0][selected_attr], branches)
    return res


def predict_one_x(tree, x, fields):
    if type(tree) is np.ndarray:
        (values, counts) = np.unique(tree, return_counts=True)
        return values[np.argmax(counts)]
    feature_name = tree[0]
    is_numeric = fields[1][fields[0].index(feature_name)]
    if is_numeric:
        value = dict(zip(fields[0], x))[feature_name]
        pivot = list(tree[1].keys())[0][1]
        if value < pivot:
            return predict_one_x(tree[1][(-np.inf, pivot)], x, fields)
        else:
            return predict_one_x(tree[1][(pivot, np.inf)], x, fields)
    else:
        branch = dict(zip(fields[0], x))[feature_name]
        return predict_one_x(tree[1][branch], x, fields)




def predict(tree, x, fields):
    y = []
    for sample in x:
        y.append(predict_one_x(tree, sample, fields))
    return y


def preprocess_data(x, fields_types):
    for i, (column, is_numeric) in enumerate(zip(x.T, fields_types)):
        if is_numeric:
            column = column.astype('float')
            inds = np.where(np.isnan(column))
            mean = np.nanmean(column)
            x[:, i][inds] = mean
        else:
            column_temp = column.astype('str')
            inds = np.where(column_temp == 'nan')
            x[:, i][inds] = column[0]
    return x


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

train_set = pd.read_csv(
    'titanic_modified.csv')

X = train_set.iloc[:, :6].values
y = train_set.iloc[:, 6].values
fields = list(train_set.columns.values)
fields_types = [False, False, True, True, True, False]
X = preprocess_data(X, fields_types)
tree = recursive_split(X, y, [fields, fields_types])
print(tree)
# print(y)
print(predict(tree, X, [fields, fields_types]) == y)


