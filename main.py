import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
import kagglehub
import math
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

####################################################################
# CONFIGURATION
####################################################################
# ALL_FRUITS_CSV = Path("fruit-prices-2023.csv")

YUGIOH_CSV = Path("yugioh-ccd-2025SEP12-163128.csv")
WALMART_CSV = Path("Walmart_Sales.csv")

OUTPUT_FOLDER = Path("output")
TOP_SALES_CSV = OUTPUT_FOLDER / Path("top_sales.csv")
TOP_TEMP_CSV = OUTPUT_FOLDER / Path("top_temp.csv")
TOP_FUEL_CSV = OUTPUT_FOLDER / Path("top_fuel.csv")
TOP_CPI_CSV = OUTPUT_FOLDER / Path("top_cpi.csv")
TOP_UNEMPLOYMENT_CSV = OUTPUT_FOLDER / Path("top_unemployment.csv")

TOP_K = 10

RNG_SEED = 0
NUM_FOLDS = 10
ALPHA_LEARNING = 1e-1
NUM_TREES = 100

####################################################################
# MAIN WALKTHRU
####################################################################
def main():
    to_wrap = pd.read_csv(WALMART_CSV)
    walmart_data = Dataset(to_wrap)
    del to_wrap

    walmart_data.to_datetime(column="Date", dayfirst=True)
    null_counts = walmart_data.isnull().any(axis=1).sum()
    print(f'There are {null_counts} entries with null values. Dropping all of those rows')

    walmart_data.dropna() #modifies in-place
    # print(walmart_data.head())

    # Aggregated each store to same week
    transformed = walmart_data.normalize()
    temp = transformed.sort_values("Date")
    # transformed_data = temp.groupby("Date").mean()
    # transformed_data_week = Dataset(transformed_data)

    # Aggregated each store to same month (ignore type-checker error on .to_timestamp())
    temp["Date"] = pd.to_datetime(temp["Date"]).dt.to_period("M")
    transformed_data = temp.groupby("Date").mean()
    transformed_data.index = transformed_data.index.to_timestamp()
    transformed_data_month = Dataset(transformed_data)

    # print(walmart_data.info())
    # print(walmart_data.head())
    # print(walmart_data.columns)

    #######################################
    #        Top K In Each Column         #
    #######################################

    # OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # cols = list(walmart_data.columns)
    # select_columns = [cols[2]] + cols[4:]
    # output_column_paths = [TOP_SALES_CSV, TOP_TEMP_CSV, TOP_FUEL_CSV, TOP_CPI_CSV, TOP_UNEMPLOYMENT_CSV]

    # for i in range(len(select_columns)):
    #     temp_data = walmart_data.nlargest(10, column=select_columns[i])
    #     temp_data.to_csv(output_column_paths[i],index=False)  

    #######################################
    #             Quick Plot              #
    #######################################
    # print(transformed_data_month.index)

    # x = transformed_data_month.data.index
    # y = transformed_data_month.data["Weekly_Sales"]
    # log_y = np.log1p(transformed_data_month.data["Weekly_Sales"])

    # plt.scatter(x,y)
    # plt.show()

    # plt.scatter(x, log_y)
    # plt.show()

    #######################################
    #           Model Testing             #
    #######################################
    model_dataset = walmart_data.model_pipeline(scale = True)
    # model_dataset["Date_continuous"] = model_dataset.index.year + (model_dataset.index.month-1)/12
    # model_dataset = model_dataset.set_index("Date_continuous")


    # print(walmart_data.head(2))
    # print(model_dataset.head())
    # print(model_dataset.columns)

    linear = LinearRegression()
    lasso = Lasso(random_state=RNG_SEED)
    ridge = Ridge(random_state=RNG_SEED)
    decision = DecisionTreeRegressor(splitter="best")
    forest = RandomForestRegressor(n_estimators = NUM_TREES, random_state = RNG_SEED)
    adaboost = AdaBoostRegressor(n_estimators = NUM_TREES, learning_rate = ALPHA_LEARNING, random_state = RNG_SEED)
    xgboost = xgb.XGBRegressor(n_estimators = NUM_TREES, learning_rate = ALPHA_LEARNING, random_state = RNG_SEED)

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

    for model_name, model in models.items():
        # perf_dict = cross_validate(dataset = transformed_data_month, model = model, folds = NUM_FOLDS, seed = RNG_SEED)
        perf_dict = single_run(dataset = model_dataset, model=model)
        performance[model_name] = perf_dict

    pd.set_option("display.float_format", "{:.4f}".format)
    perf_df = pd.DataFrame(performance).T 

    print(perf_df)

    #######################################
    #    Plot Feature Comparison Plots    #
    #######################################

    # cols = ['Store', 'Date', 'Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    # cols.remove("Store")
    # cols.remove("Date")
    # cols.remove("Weekly_Sales")
    # cols.remove("Holiday_Flag")

    # for col in cols:
    #     two_lines_plot(dataset=transformed_data_month, col_one="Weekly_Sales", col_two=col)

    #######################################
    #      Plot Correlation Matrices      #
    #######################################

    # print(transformed_data_month)

    # correlation_heatmap(dataset=walmart_data)
    # correlation_heatmap(dataset=transformed_data_week)
    # correlation_heatmap(dataset=transformed_data_month)


    #######################################
    #          Plot Scatterplot          #
    #######################################
    
    # start, end = 1, 1
    # for i in range(start, end + 1):
    #     display_plot(
    #         data=walmart_data,
    #         store_num=i,
    #         holidays=True, 
    #         options=0
    #     )

    #######################################
    #             Plot Splines            #
    #######################################

    # start, end = 1, 1
    # for i in range(start, end + 1):
    #     display_plot(
    #         data=walmart_data, 
    #         store_num=i,
    #         holidays=False, 
    #         options = 1
    #     )

    #######################################
    #     Plot Categorical Pie Charts     #
    #######################################

    # display_pie_chart(data=walmart_data, column="Holiday_Flag", threshold = 2.8e-2, title="Distribution of Holiday Weeks")


if __name__ == "__main__":
    start_time = time.time()
    main()
    time_passed = time.time() - start_time
    print(f"Time to run: {time_passed}")