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
TOP_SALES_CSV = OUTPUT_FOLDER / Path("top_sales.csv")
TOP_TEMP_CSV = OUTPUT_FOLDER / Path("top_temp.csv")
TOP_FUEL_CSV = OUTPUT_FOLDER / Path("top_fuel.csv")
TOP_CPI_CSV = OUTPUT_FOLDER / Path("top_cpi.csv")
TOP_UNEMPLOYMENT_CSV = OUTPUT_FOLDER / Path("top_unemployment.csv")

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
    # # correlation_heatmap(dataset=transformed_data_week)
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

    #######################################
    #           Model Testing             #
    #######################################
    # run_models(dataset=walmart_data, k_folds=NUM_FOLDS, n_trees=NUM_TREES, seed=RNG_SEED, alpha_learning=ALPHA_LEARNING)

if __name__ == "__main__":
    start_time = time.time()
    main()
    time_passed = time.time() - start_time
    print(f"Time to run: {time_passed}")