import numpy as np

from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit, cross_val_score
from sklearn.base import BaseEstimator, RegressorMixin

from typing import Callable
from data_obj import Dataset

Metric = Callable[..., float]

def cross_validate(x, y, model: BaseEstimator, folds: int):
    # model = Ridge(alpha=1.0)
    # data = dataset.data
    tscv = TimeSeriesSplit(n_splits = folds)

    # cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_root_mean_squared_error')

    # if shuffle:
    #     k_folds = KFold(n_splits=folds, shuffle=True, random_state=seed) # TIME SERIES DONT SHUFFLE
    # else:
    #     k_folds = KFold(n_splits=folds)
    # val = False

    mse_list = []
    mae_list = []
    r_squared_list = []

    for train_idx, test_idx in tscv.split(x):
        # y = data["Weekly_Sales"]
        # x = data.drop("Weekly_Sales", axis=1)
        # print(y.head())
        # print(x.head())

        x_train, y_train = x.iloc[train_idx], y.iloc[train_idx]
        x_test, y_test = x.iloc[test_idx], y.iloc[test_idx]
        # a = np.isnan(x_test).sum().sum()
        # b = np.isnan(x_train).sum().sum()
        # if a > 0 or b > 0:
        #     val = True

        model.fit(x_train, y_train)

        predicted = model.predict(x_test)
        # c = np.isnan(predicted).sum().sum()
        # if c > 0:
        #     val = True
        #     print(x_test)
        # print(val)

        mse = mean_squared_error(y_test, predicted)
        mae = mean_absolute_error(y_test, predicted)
        r2 = r2_score(y_test, predicted)

        mse_list.append(mse)
        mae_list.append(mae)
        r_squared_list.append(r2)

    avg_mse = sum(mse_list) / len(mse_list)
    avg_mae = sum(mae_list) / len(mae_list)
    avg_r_squared = sum(r_squared_list) / len(r_squared_list)
    # print( mse_list)
    # print( r_squared_list)
    
    return {"MSE": avg_mse, "MAE": avg_mae, "R2": avg_r_squared}


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