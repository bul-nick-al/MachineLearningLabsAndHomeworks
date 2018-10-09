import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def read_and_clean_data():
    df = pd.read_csv('ks.csv', encoding="ISO-8859-1")
    df['deadline '] = pd.to_datetime(df['deadline '], errors='coerce')
    df['launched '] = pd.to_datetime(df['launched '], errors='coerce')
    df = df.dropna(subset=['deadline '])
    df = df.dropna(axis=1, thresh=100)
    df = df[(df['state '] == 'failed') | (df['state '] == 'successful')]
    df['year'] = df['launched '].map(lambda x: x.year)
    df['duration'] = df['deadline '].map(lambda x: x.year*365 + x.month*30 + x.day) - df['launched ']\
        .map(lambda x: x.year * 365 + x.month*30 + x.day)
    df['exchange_rates'] = (pd.to_numeric(df['usd pledged ']) / pd.to_numeric(df['pledged '])).fillna(value=1)
    df['exchange_rates'] = df['exchange_rates'].replace(0, 1)
    df['goal '] = pd.to_numeric(df['goal '])*df['exchange_rates']
    df = pd.get_dummies(df, columns=['state '])
    df = pd.get_dummies(df, columns=['currency '])
    df = pd.get_dummies(df, columns=['main_category '])

    df = df.drop(['ID ', 'name ', 'category ', 'pledged ', 'launched ', 'deadline ', 'exchange_rates',
                  'country ', 'backers '], axis=1)
    return df


df = read_and_clean_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.55)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)