import pandas as pd
import random
from matplotlib import pyplot as plt


def split_to_sets(df: pd.DataFrame, fraction: float) -> (pd.DataFrame, pd.DataFrame):
    df = df.sample(frac=1)
    delimiter_index = int(len(df) * fraction)
    return df[: delimiter_index].reset_index(drop=True), df[delimiter_index:].reset_index(drop=True)


df = pd.read_csv("/Users/nicholas/Downloads/datasets/iris.csv")


a = split_to_sets(df, 0.5)



