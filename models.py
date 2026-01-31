import numpy as np
import time

from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit, cross_val_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

from typing import Callable
from data_obj import Dataset

Metric = Callable[..., float]

def cross_validate(x, y, model: BaseEstimator, folds: int, validation_size = 0.2):
    start_time = time.time()
    x_training, x_val, y_training, y_val = train_test_split(x, y, test_size = validation_size, shuffle=False)
    tscv = TimeSeriesSplit(n_splits = folds)

    mse_list = []
    mae_list = []
    r_squared_list = []

    for train_idx, test_idx in tscv.split(x_training):
        x_train, y_train = x_training.iloc[train_idx], y_training.iloc[train_idx]
        x_test, y_test = x_training.iloc[test_idx], y_training.iloc[test_idx]

        model.fit(x_train, y_train)
        predicted = model.predict(x_test)

        mse = mean_squared_error(y_test, predicted)
        mae = mean_absolute_error(y_test, predicted)
        r2 = r2_score(y_test, predicted)

        mse_list.append(mse)
        mae_list.append(mae)
        r_squared_list.append(r2)

    avg_mse = sum(mse_list) / len(mse_list)
    avg_mae = sum(mae_list) / len(mae_list)
    avg_r_squared = sum(r_squared_list) / len(r_squared_list)
    # train_metrics = {"MSE": avg_mse, "MAE": avg_mae, "R2": avg_r_squared}

    predicted = model.predict(x_val)
    mse = mean_squared_error(y_val, predicted)
    mae = mean_absolute_error(y_val, predicted)
    r2 = r2_score(y_val, predicted)
    total_time = time.time() - start_time

    # val_metrics = {"MSE": mse, "MAE": mae, "R2": r2}

    return {
        "Train_MSE": avg_mse, 
        "Train_MAE": avg_mae, 
        "Train_R2": avg_r_squared, 
        "Validation_MSE": mse, 
        "Validation_MAE": mae, 
        "Validation_R2": r2,
        "Runtime": total_time
    }
    # return {"train_metrics": train_metrics, "val_metrics": val_metrics}


def single_run(x, y, model: BaseEstimator, test_size = 0.2):
    # data = dataset.data
    # data = data.sort_values("Date_Continuous")  # ensure chronological order

    # y = data["Log_Weekly_Sales"]
    # x = data.drop("Log_Weekly_Sales", axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, shuffle=False)

    model.fit(x_train, y_train)

    predicted = model.predict(x_test)

    mse = mean_squared_error(y_test, predicted)
    mae = mean_absolute_error(y_test, predicted)
    r2 = r2_score(y_test, predicted)

    return {"MSE": mse, "MAE": mae, "R2": r2}