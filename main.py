import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
import kagglehub
import math

from scipy.interpolate import UnivariateSpline, BSpline
from pathlib import Path

from matplotlib.ticker import FuncFormatter, MultipleLocator
from data_obj import *
from data_display import *

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

####################################################################
# MAIN WALKTHRU
####################################################################
def main():
    to_wrap = pd.read_csv(WALMART_CSV)
    walmart_data = Dataset(to_wrap)
    walmart_data.to_datetime(column="Date", dayfirst=True)
    del to_wrap

    null_counts = walmart_data.isnull().any(axis=1).sum()
    print(f'There are {null_counts} entries with null values. Dropping all of those rows')

    walmart_data.dropna() #modifies in-place

    # print(walmart_data.info())
    # print(walmart_data.head())
    # print(walmart_data.columns)

    # ['Store', 'Date', 'Weekly_Sales', 'Holiday_Flag', 'Temperature',
    #    'Fuel_Price', 'CPI', 'Unemployment']

    # OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # cols = list(walmart_data.columns)
    # select_columns = [cols[2]] + cols[4:]
    # output_column_paths = [TOP_SALES_CSV, TOP_TEMP_CSV, TOP_FUEL_CSV, TOP_CPI_CSV, TOP_UNEMPLOYMENT_CSV]

    # for i in range(len(select_columns)):
    #     temp_data = walmart_data.nlargest(10, column=select_columns[i])
    #     temp_data.to_csv(output_column_paths[i],index=False)

    ###############################
    #      Plot Scatterplots      #
    ###############################

    # start, end = 1, 1
    # for i in range(start, end + 1):
    #     display_plot(
    #         data=walmart_data, 
    #         store_num=i,
    #         holidays=True, 
    #         options=0
    #     )

    ###############################
    #        Plot Splines         #
    ###############################

    # start, end = 1, 1
    # for i in range(start, end + 1):
    #     display_plot(
    #         data=walmart_data, 
    #         store_num=i,
    #         holidays=False, 
    #         options = 1
    #     )

    ###############################
    # Plot Categorical Pie Charts #
    ###############################

    # display_pie_chart(data=walmart_data, column="Holiday_Flag", threshold = 2.8e-2, title="Distribution of Holiday Weeks")


if __name__ == "__main__":
    start_time = time.time()
    main()
    time_passed = time.time() - start_time
    print(f"Time to run: {time_passed}")