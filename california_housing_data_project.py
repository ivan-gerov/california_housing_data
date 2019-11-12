import os
import pandas as pd
import numpy as np
import tarfile
from six.moves import urllib

# 1. Getting the Data

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


# fetch_housing_data()


def load_housing_data(housing_path=HOUSING_PATH):

    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


housing = load_housing_data()
# housing.head()
# housing.info()
# housing['ocean_proximity'].value_counts()

import matplotlib.pyplot as plt

# housing.hist(bins= 50, figsize= (20, 15)) # Figsize accepts a (height, length) tuple
# plt.show()

# 2. Splitting the Data into a Train and Test sets

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# print('Train set:', len(train_set), "+ Test set:", len(test_set))
# housing['median_income'].hist()
# plt.show()

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

# Stratified Sampling

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)

# for column in strat_train_set.columns:
#     print(column)

housing = strat_train_set.copy()

# Simple Exploratory Data Analysis with Matplotlib

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

# Creating new columns, which may be more informative than
# the ones we already have

housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# Preparing the Data

# Filling Missing Values in Numerical Attributes with the Median of the Attribute

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
housing_num = housing.drop(
    "ocean_proximity", axis=1
)  # Separating the test attribute from the Nums
imputer.fit(housing_num)

# imputer.statistics_

x = imputer.transform(housing_num)  # NumPy Array

housing_tr = pd.DataFrame(
    x, columns=housing_num.columns
)  # putting it all back in a DataFrame

# Recoding the Categorical (Text) Attributes with sklearn.preprocessing.OrdinalEncoder

housing_cat = housing[["ocean_proximity"]]
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]
ordinal_encoder.categories_

# Recoding the Categorial (Text) Attributes with sklearn.preprocessing.OneHotEncoder

from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot

# Writing Custome Transformers with Sklearn (sklearn.base.BaseEstimator|TransformerMixin)

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or *kwargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[
                X, rooms_per_household, population_per_household, bedrooms_per_room
            ]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

# Writing our own Custom Data Pipelines with sklearn.pipeline.Pipeline

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# First creating a numerical attribute pipeline
num_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("attribs_adder", CombinedAttributesAdder()),
        ("std_scaler", StandardScaler()),
    ]
)

housing_num_tr = num_pipeline.fit_transform(housing_num)
housing_num_tr

# Combining our Custom Data Pipelines and Custom Transformers in a ColumnTransformer (Combined Pipeline)


from sklearn.compose import ColumnTransformer


# ColumnTransformer only takes in lists with the pd.DataFrame columns that are
# to be transformed

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]


num_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("attribs_adder", CombinedAttributesAdder()),
        ("std_scaler", StandardScaler()),
    ]
)

cat_pipeline = Pipeline([("1hot_encoder", OneHotEncoder()),])


full_pipeline = ColumnTransformer(
    [("num", num_pipeline, num_attribs), ("cat", cat_pipeline, cat_attribs),]
)
housing_prepared = full_pipeline.fit_transform(housing)

# Selecting and Training a Model

# 1. Linear Regression

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels", list(some_labels))
from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
housing_labels.describe()
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
