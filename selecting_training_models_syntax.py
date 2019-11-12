
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np 
from california_housing_data_project import (housing, housing_labels,
                                             housing_prepared,
                                             full_pipeline)

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

# print("Predictions:", lin_reg.predict(some_data_prepared))

housing_predictions = lin_reg.predict(some_data_prepared)

lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)

# print(lin_rmse)

