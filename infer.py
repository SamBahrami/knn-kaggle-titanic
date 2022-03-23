import pandas as pd
import matplotlib.pyplot as plt
from yaml import safe_load
from typing import List


def load_datasets(data_params):
    train_data = pd.read_csv(data_params["train"])
    test_data = pd.read_csv(data_params["test"])
    return train_data, test_data


def normalize_df_column(df, column: str):
    """Normalize a column so values are between 0 and 1"""
    pass


def replace_column_with_numbers(df, column: str):
    """A column of string values, take the unique values and replace them
    with integers instead"""
    pass


def remove_columns(df, columns: List[str]):
    pass


def infer_on_test():
    pass


with open("params.yml", "r") as file:
    params = safe_load(file)
train, test = load_datasets(params["data"])

pass
