import re
import time
import math

import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import kagglehub
import seaborn as ss
import xgboost as xgb

from collections import defaultdict
from pathlib import Path
from scipy.interpolate import UnivariateSpline, BSpline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit, cross_val_score
from matplotlib.ticker import FuncFormatter, MultipleLocator

from data_obj import *
from data_display import *
from models import *

def run_models(dataset: Dataset, k_folds: int, n_trees: int, seed: int, alpha_learning: float, validate_size = 0.2):
    model_dataset = dataset.model_pipeline(scale = True)
    data = model_dataset.data
    data = data.sort_values("Date_Continuous")
    columns = data.columns
    store_encode_cols = [col for col in columns if re.search(r"Store_\d+", col)]

    # print(walmart_data.head(2))
    # print(model_dataset.head())
    # print(model_dataset.columns)

    linear = LinearRegression()
    lasso = Lasso(random_state=seed)
    ridge = Ridge(random_state=seed)
    decision = DecisionTreeRegressor(splitter="best")
    forest = RandomForestRegressor(n_estimators = n_trees, random_state = seed)
    adaboost = AdaBoostRegressor(n_estimators = n_trees, learning_rate = alpha_learning, random_state = seed)
    xgboost = xgb.XGBRegressor(n_estimators = n_trees, learning_rate = alpha_learning, random_state = seed)

    models = {
        "Linear": linear,
        "Lasso": lasso,
        "Ridge": ridge,
        "Decision": decision,
        "Random Forest": forest,
        "Ada Boost Regression": adaboost,
        "XG Boost Regression": xgboost
    }

    performance = {}

    # col = [
    #     "Cos_Month", 
    #     "Sin_Month", 
    #     "Peak_Season", 
    #     'Sales_Lag_One', 
    #     'Sales_Lag_Two', 
    #     "Log_Weekly_Sales",
    #     "Date",
    #     "month",
    #     "Weekly_Sales",
    #     "Store"
    # ]
    # # col.extend(store_encode_cols)
    # x = data.drop(columns=col, axis=1)
    # y = data["Log_Weekly_Sales"]
    
    # print("\nBasic features")
    # print("-" * 64)
    # for model_name, model in models.items():
    #     perf_dict = cross_validate(x=x, y=y, model = model, folds = k_folds)
    #     # perf_dict = single_run(x=x, y=y, model=model)
    #     performance[model_name] = perf_dict

    # pd.set_option("display.float_format", "{:.4f}".format)
    # perf_df = pd.DataFrame(performance).T 

    # print(perf_df)

    #####################################
    col = [
        "Date", 
        'month', 
        "Weekly_Sales", 
        "Store",
        "Log_Weekly_Sales"
    ]
    x = data.drop(columns=col, axis=1)
    y = data["Log_Weekly_Sales"]
    
    print("\nAll new features")
    print("-" * 64)
    for model_name, model in models.items():
        perf_dict = cross_validate(x=x, y=y, model = model, folds = k_folds, validation_size = validate_size)
        # perf_dict = single_run(x=x, y=y, model=model)
        performance[model_name] = perf_dict

    pd.set_option("display.float_format", "{:.4f}".format)
    perf_df = pd.DataFrame(performance).T 

    print(perf_df)
    return perf_df