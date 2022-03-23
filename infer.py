import pandas as pd
import matplotlib.pyplot as plt
from yaml import safe_load
import numpy as np


def take_first(elem):
    return elem[0]


class knn:
    def __init__(self, train, target, params):
        self.train_data = train
        # self.data_columns = list(train.columns)
        self.train_targets = target
        self.k = params["k"]

    @staticmethod
    def _distance(list_one, list_two):
        dist = np.linalg.norm(list_one - list_two)
        return dist

    def infer(self, datapoint):
        # raw_datapoint = np.array(datapoint[self.data_columns])
        raw_datapoint = datapoint
        closest_k_distances = []
        for i, raw_train_data in enumerate(self.train_data):
            distance = self._distance(raw_datapoint, raw_train_data)
            closest_k_distances.sort(key=take_first)
            if len(closest_k_distances) < self.k:
                closest_k_distances.append((distance, self.train_targets[i]))
                continue
            if distance < closest_k_distances[-1][0]:
                closest_k_distances[-1] = (distance, self.train_targets[i])
        sum_of_targets = 0
        for item in closest_k_distances:
            sum_of_targets += item[1]
        if sum_of_targets < (self.k / 2.0):
            return 0
        return 1


def load_datasets(data_params, knn_features):
    train_data = pd.read_csv(data_params["train"])
    train_data = train_data[knn_features + ["Survived"]]
    # For simplicity
    train_data["Age"].fillna(train_data["Age"].mean(), inplace=True)
    train_data.dropna(inplace=True)
    train_target = train_data["Survived"]
    train_data = train_data[knn_features]
    test_data = pd.read_csv(data_params["test"])
    return train_data, train_target, test_data


def replace_column_with_numbers(df, column: str):
    """A column of string values, take the unique values and replace them
    with integers instead"""
    unique_values = sorted(df[column].unique())
    for i in range(0, len(unique_values)):
        # you can probably do this in one call instead of len(unique_values)
        df[column].replace(unique_values[i], i * 1.0, inplace=True)
    return


def normalize_df_columns(train, test):
    """Normalize all columns so values are between 0 and 1"""
    train_maxes = train.max()
    train = train / train_maxes
    test = test / train_maxes
    return train, test


def display_train_data(train, targets):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    target_classes = list(targets.unique())
    markers = ["o", "^", "x"]
    for i, target_class in enumerate(target_classes):
        current_class_data = train[targets == target_class]
        ax.scatter(
            current_class_data["Age"],
            current_class_data["Fare"],
            *current_class_data["Age"] * current_class_data["Pclass"],
            marker=markers[i % len(markers)],
        )
    ax.set_xlabel("Age")
    ax.set_ylabel("Fare")
    ax.set_zlabel("Pclass*Age")
    plt.show()


def convert_data_to_numpy(train, test, target):
    return np.array(train), np.array(test), np.array(target)


with open("params.yml", "r") as file:
    params = safe_load(file)

knn_params = params["knn_features"]
train, target, raw_test = load_datasets(params["data"], knn_params["features"])
test = raw_test[knn_params["features"]]

for textual_feature in knn_params["text_features"]:
    replace_column_with_numbers(train, textual_feature)
    replace_column_with_numbers(test, textual_feature)

train, test = normalize_df_columns(train, test)
display_train_data(train, target)

train, test, target = convert_data_to_numpy(train, test, target)
knn_model = knn(train, target, params["knn"])
results = []

for row in test:
    results.append(knn_model.infer(row))
raw_test["Survived"] = results
raw_test[["PassengerId", "Survived"]].to_csv("Solution.csv", index=False)
pass
