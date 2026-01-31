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
from run_models import *

####################################################################
# CONFIGURATION
####################################################################
# ALL_FRUITS_CSV = Path("fruit-prices-2023.csv")
# YUGIOH_CSV = Path("yugioh-ccd-2025SEP12-163128.csv")

WALMART_CSV = Path("Walmart_Sales.csv")

OUTPUT_FOLDER = Path("output")
STORE_FOLDER = OUTPUT_FOLDER / Path("stores")

TOP_K = 10

RNG_SEED = 314
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
    # print(len(walmart_data))

    null_counts = walmart_data.isnull().any(axis=1).sum()
    num_duplicates = walmart_data.data.duplicated().sum()
    print(f'There are {null_counts} entries with null values. Dropping all of those rows')
    print(f"There are {num_duplicates} entries with duplicate rows. Dropping all of those rows")

    walmart_data.dropna() 
    walmart_data.drop_duplicates()

    # Aggregated each store to same week
    # transformed = walmart_data.normalize()
    # temp = transformed.sort_values("Date")
    # transformed_data = temp.groupby("Date").mean()
    # transformed_data_week = Dataset(transformed_data)

    # Aggregated each store to same month (ignore type-checker error on .to_timestamp())
    transformed = walmart_data.normalize() #normalize
    temp = transformed.sort_values("Date") #sort
    temp["Date"] = pd.to_datetime(temp["Date"]).dt.to_period("M") #drop day
    transformed_data = temp.groupby("Date").mean() #average all features in same month
    transformed_data.index = transformed_data.index.to_timestamp() #back to timestamp for plot
    transformed_data_month = Dataset(transformed_data)

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    STORE_FOLDER.mkdir(parents=True, exist_ok=True)

    # print(walmart_data.columns)

    #######################################
    #        Top K In Each Column         #
    #######################################
    # top_sales_csv = OUTPUT_FOLDER / Path("top_sales.csv")
    # top_temp_csv = OUTPUT_FOLDER / Path("top_temp.csv")
    # top_fuel_csv = OUTPUT_FOLDER / Path("top_fuel.csv")
    # top_cpi_csv = OUTPUT_FOLDER / Path("top_cpi.csv")
    # top_unemployment_csv = OUTPUT_FOLDER / Path("top_unemployment.csv")


    # cols = list(walmart_data.columns)
    # select_columns = [cols[2]] + cols[4:]
    # output_column_paths = [top_sales_csv, top_temp_csv, top_fuel_csv, top_cpi_csv, top_unemployment_csv]

    # for i in range(len(select_columns)):
    #     temp_data = walmart_data.nlargest(10, column=select_columns[i])
    #     temp_data.to_csv(output_column_paths[i],index=False)  

   
    #######################################
    #    Plot Feature Comparison Plots    #
    #######################################

    # cols = ['Store', 'Date', 'Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    # cols.remove("Store")
    # cols.remove("Date")
    # cols.remove("Weekly_Sales")
    # cols.remove("Holiday_Flag")

    # for col in cols:
    #     title = f"Weekly_Sales_&_{col}_comparison_plot.png"
    #     output_file = OUTPUT_FOLDER / title
    #     two_lines_plot(dataset=transformed_data_month, col_one="Weekly_Sales", col_two=col, output_file=output_file)

    #######################################
    #      Plot Correlation Matrices      #
    #######################################

    # print(transformed_data_month)

    # corr_whole = OUTPUT_FOLDER / Path("correlation_whole_dataset.png")
    # corr_avg_month = OUTPUT_FOLDER / Path("correlation_average_month.png")

    # correlation_heatmap(dataset=walmart_data, output_file=corr_whole)
    # # correlation_heatmap(dataset=transformed_data_week)
    # correlation_heatmap(dataset=transformed_data_month, output_file= corr_avg_month)

    #######################################
    #          Plot Scatterplot          #
    #######################################

    # start, end = 1, 45
    # for i in range(start, end + 1):
    #     output_file = STORE_FOLDER / Path(f"store_{i}")
        
    #     display_plot(
    #         data=walmart_data,
    #         store_num=i, 
    #         output_file=output_file,
    #         options=0
    #     )

    #######################################
    #             Plot Splines            #
    #######################################
    # start, end = 1, 49
    # for i in range(start, end + 1):
    #     output_file = STORE_FOLDER / Path(f"store_spline_{i}")

    #     display_plot(
    #         data=walmart_data, 
    #         store_num=i,
    #         output_file=output_file,
    #         options = 1
    #     )

    #######################################
    #     Plot Categorical Pie Charts     #
    #######################################
    # output_file = OUTPUT_FOLDER / Path("holiday_flag_pie_chart")

    # display_pie_chart(data=walmart_data, column="Holiday_Flag", threshold = 2.8e-2, title="Distribution of Holiday Weeks", output_file=output_file)

    #######################################
    #           Model Testing             #
    #######################################
    output_file = OUTPUT_FOLDER / Path("all_feature_models_metrics.csv")

    perf_df = run_models(dataset=walmart_data, k_folds=NUM_FOLDS, n_trees=NUM_TREES, seed=RNG_SEED, alpha_learning=ALPHA_LEARNING)
    perf_df = perf_df.reset_index()
    perf_df = perf_df.rename(columns={"index": "Model Name"})
    perf_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    start_time = time.time()
    main()
    time_passed = time.time() - start_time
    print(f"Time to run: {time_passed}")