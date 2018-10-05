import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def plot_corr_matrix(corr):
    names = corr.columns.values
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr, vmin=-1, vmax=1)
    ticks = np.arange(0, 15, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.show()


def read_and_clean_data():
    df = pd.read_csv('avocado.csv')
    df = pd.get_dummies(df, columns=['type'])
    df = df.drop([df.columns[0]], axis=1)
    months = df['Date'].str.extract(r'....-?(\d{2})-..')
    df = df.drop(['Date'], axis=1)
    df['months'] = pd.to_numeric(months[months.columns[0]])
    df['yearMonth'] = df['year'] * 12 + df['months']
    return df


df = read_and_clean_data()
split_data = {}
for region in df.region.unique():
    split_data[region] = (df[df['region'] == region])[['yearMonth', 'AveragePrice']]
corr = df.corr()
plot_corr_matrix(corr)
evaluation = {}

for region, region_data in split_data.items():
    lr = LinearRegression()
    X = region_data.drop('AveragePrice', axis=1)
    y = region_data.drop('yearMonth', axis=1)
    lr.fit(X, y)
    y_predict = lr.predict(pd.DataFrame([2020*12+6]))
    evaluation[region] = y_predict[0][0]

max_score = 0
best_region = ''
for region, score in evaluation.items():
    if score > max_score:
        max_score = score
        best_region = region
print(best_region)







# X = df.drop(['AveragePrice', 'Date', df.columns[0]], axis=1)
# y = df['AveragePrice']
# lm = LinearRegression()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# lm.fit(X_train, y_train)
# print(lm.score(X_test, y_test))
# plt.scatter(X_train["year"], y_train)
# plt.legend()
# plt.show()


# Correction Matrix Plot

