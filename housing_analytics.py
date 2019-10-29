
import os
import pandas as pd
import numpy as np
from fetch_data import HOUSING_PATH, fetch_housing_data


def load_housing_data(housing_path= HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)

housing = load_housing_data()

def split_train_test(data, test_ration):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

import hashlib

def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash= hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_ : test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index()

split_train_test_by_id(housing_with_id,
                        test_ratio= 0.2,
                        id_column = 'index')

def test_set_check2(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[0] #< 256 * test_ratio
                    
print(test_set_check2(np.int64(102), 0.2, hashlib.md5))