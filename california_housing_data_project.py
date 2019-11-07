import os
import pandas as pd
import numpy as np
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
    Requires a the data origin path (housing_url) and the data destination path (housing_path).
    1. Checks if the housing_path exists under our root dir. If not, creates it.
    2. Downloads the dataset (which is in .tgz format) to the root/housing_path.
    3. Extracts all files from the dataset to the root/housing_path folder.
    """
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


fetch_housing_data()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


housing = load_housing_data()


housing.head()
housing.info()
housing["ocean_proximity"].value_counts()


import matplotlib.pyplot as plt

housing.hist(bins=50, figsize=(20, 15))  # Figsize accepts a (height, length) tuple

plt.show()


from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

print("Train set:", len(train_set), "+ Test set:", len(test_set))

housing["median_income"].hist()
plt.show()


housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)

for column in strat_train_set.columns:
    print(column)

housing = strat_train_set.copy()

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.2, figsize=(10, 8))

plt.show()

housing.plot(
    kind="scatter",
    x="longitude",
    y="latitude",
    alpha=0.4,
    s=housing["population"] / 100,
    label="population",
    c="median_house_value",
    cmap=plt.get_cmap("jet"),
    colorbar=True,
    figsize=(10, 8),
)
plt.show()
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
from pandas.plotting import scatter_matrix

attributes = [
    "median_house_value",
    "median_income",
    "total_rooms",
    "housing_median_age",
]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()
housing.plot(
    kind="scatter",
    x="median_income",
    y="median_house_value",
    alpha=0.6,
    figsize=(10, 8),
)
plt.show()

housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]


corr_matrix = housing.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

housing.dropna(subset= ['total_bedrooms']) #option 1
housing.drop('total_bedrooms', axis=1)     #option 2
median = housing['total_bedrooms'].median()
housing['total_bedrooms'].fillna(median)

