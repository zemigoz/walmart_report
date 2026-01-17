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
ALL_FRUITS_CSV = Path("fruit-prices-2023.csv")

YUGIOH_CSV = Path("yugioh-ccd-2025SEP12-163128.csv")
WALMART_CSV = Path("Walmart_Sales.csv")

####################################################################
# MAIN WALKTHRU
####################################################################
def main():
    # to_wrap = pd.read_csv(YUGIOH_CSV)
    # yugioh_data = Dataset(to_wrap)

    # print(yugioh_data.info())
    # print(yugioh_data.head())
    # print(yugioh_data.columns)
    # ['Unnamed: 0', 'name', 'description', 'set_id', 'rarity', 'price',
    #    'volatility', 'type', 'sub_type', 'attribute', 'rank', 'attack',
    #    'defense', 'set_name', 'set_release', 'name_official', 'index',
    #    'index_market', 'join_id']

    # print(yugioh_data.rarity_distribution("type"))

    to_wrap = pd.read_csv(WALMART_CSV)
    walmart_data = Dataset(to_wrap)
    walmart_data.to_datetime(column="Date", dayfirst=True)

    # print(walmart_data.info())
    # print(walmart_data.head())
    # print(walmart_data.columns)

    # ['Store', 'Date', 'Weekly_Sales', 'Holiday_Flag', 'Temperature',
    #    'Fuel_Price', 'CPI', 'Unemployment']

    


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

    # display_pie_chart(data=yugioh_data, column="rarity", threshold = 2.8e-2, title="Distribution of Yu-Gi-Oh! Rarities")
    # display_pie_chart(data=yugioh_data, column="volatility", threshold = 2.8e-2, title="Distribution of Yu-Gi-Oh! Volatility in the market")
    # display_pie_chart(data=yugioh_data, column="type", threshold = 2.8e-5, title="Distribution of Yu-Gi-Oh! Card Types")
    # display_pie_chart(data=yugioh_data, column="rank", threshold = 3.3e-2, title="Distribution of Yu-Gi-Oh! Card Ranks")


if __name__ == "__main__":
    start_time = time.time()
    main()
    time_passed = time.time() - start_time
    print(f"Time to run: {time_passed}")