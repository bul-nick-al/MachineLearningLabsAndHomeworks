import numpy as np
import pandas as pd
# code is based on this resource - http://gabrielelanaro.github.io/blog/2016/03/03/decision-trees.html

def partition(a):
    return {c: (a == c).nonzero()[0] for c in np.unique(a)}


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


def information_gain_ratio(y, x):
    return information_gain(y,x)/intrinsic_value(y,x)


def intrinsic_value(y, x):
    # We partition x, according to attribute values x_i
    val, counts = np.unique(x, return_counts=True)
    freqs = counts.astype('float') / len(x)

    res = 0

    # We calculate a weighted average of the entropy
    for p, v in zip(freqs, val):
        res -= p * np.log2(p)
    return res



def is_pure(s):
    return len(set(s)) == 1


def recursive_split(x, y, fields):
    # If there could be no split, just return the original set
    if is_pure(y) or len(y) == 0:
        return y

    for x_attr in x.T:
        print(information_gain(y, x_attr), information_gain_ratio(y, x_attr))

    # We get attribute that gives the highest information gain
    gain = np.array([information_gain(y, x_attr) for x_attr in x.T])
    selected_attr = np.argmax(gain)

    # If there's no gain at all, nothing has to be done, just return the original set
    if np.all(gain < 1e-6):
        return y

    # We split using the selected attribute
    sets = partition(x[:, selected_attr])

    res = {}
    for k, v in sets.items():
        y_subset = y.take(v, axis=0)
        x_subset = x.take(v, axis=0)

        res["{} = {}".format(fields[selected_attr], k)] = recursive_split(
            x_subset, y_subset, fields)

    return res


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

train_set = pd.read_csv(
    'https://raw.githubusercontent.com/sjwhitworth/golearn/master/examples/datasets/tennis.csv')

X = train_set.iloc[:, :4].values
y = train_set.iloc[:, 4].values

fields = list(train_set.columns.values)
tree = recursive_split(X, y, fields)
print(tree)
