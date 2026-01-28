import re

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sb

from scipy.interpolate import UnivariateSpline, BSpline
from matplotlib.ticker import FuncFormatter, MultipleLocator
from typing import TypedDict

from data_obj import *

def display_pie_chart(data: Dataset, column: str, threshold: float, title: str, output_file = None) -> None:
    rarity_dist = data.rarity_distribution(column=column)

    # threshold = 2.8e-2 
    total = rarity_dist.sum()
    below_threshold = rarity_dist[rarity_dist / total < threshold]
    at_least_threshold = rarity_dist[rarity_dist / total >= threshold]
    # below_threshold = [x for x in below_threshold if ]
    # print(type(below_threshold))

    threshold_dist = pd.concat([at_least_threshold, pd.Series({"Other": below_threshold.sum()})])
    threshold_dist = threshold_dist[threshold_dist > 0]

    plt.figure(figsize=(6,6))
    plt.pie(threshold_dist, labels=["No Holiday Week", "Holiday Week"], autopct='%1.1f%%', startangle=90)
    plt.title(title)

    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()

def display_plot(
    data: Dataset, 
    store_num: int,
    output_file=None, 
    options=0
):
    """
    Option 0 plots a store over time vs Weekly_Sales. Option 1 adds a B-spline. Option 2 splits Holiday_Flag feature. Option 3 applies smoothing
    by averaging nearby points. Option 4 is Option 0 and 1.

    Args:
        data (Dataset): dataset to use for walmart data
        store_num (int): store number of choice to plot
        options (int, optional): selects a plot method. Defaults to 0.

    """
        
    store_data = data[data["Store"] == (store_num)]
    store_data = store_data.sort_values("Date")
    x = store_data["Date"].map(pd.Timestamp.toordinal)
    y = store_data["Weekly_Sales"].values 

    # holiday_one["Weekly_Sales_Smooth"] = holiday_one["Weekly_Sales"].rolling(window=5, center=True).mean()
    # holiday_zero["Weekly_Sales_Smooth"] = holiday_zero["Weekly_Sales"].rolling(window=5, center=True).mean()

    plt.figure(figsize=(8,8))
    plt.xlabel("Date")
    plt.ylabel("Weekly Sales (in USD Millions)")
    plt.title("Weekly Sales Over Time Scatter Plot")
    
    if options == 0: # Regular plot
        plt.scatter(store_data["Date"], store_data["Weekly_Sales"], color='darkgray', s=20)

    elif options == 1: # With spline
        spline = UnivariateSpline(x, y, k=3, s=1e10)
        xs = np.linspace(x.min(), x.max(), 500)
        ys = spline(xs)

        xs_dates = [pd.Timestamp.fromordinal(int(d)) for d in xs]
        plt.plot(np.array(xs_dates), ys, color='orange', label='B-spline')

    elif options == 2: # Holiday split scatter
        holiday_zero = store_data[store_data["Holiday_Flag"] == 0]
        holiday_one = store_data[store_data["Holiday_Flag"] == 1]

        plt.scatter(holiday_zero["Date"], holiday_zero["Weekly_Sales"], color='skyblue', s=20)
        plt.scatter(holiday_one["Date"], holiday_one["Weekly_Sales"], color='salmon', s=20)

    elif options == 3: # Rolled/Smooth scatter
        store_data["Weekly_Sales_Smooth"] = store_data["Weekly_Sales"].rolling(window=5, center=True).mean()
        y = store_data["Weekly_Sales_Smooth"]

        plt.scatter(store_data["Date"], store_data["Weekly_Sales_Smooth"], color='dimgray', s=20)
        plt.title("Weekly Sales (Smoothed) Over Time Scatter Plot")

    elif options == 4: # Regular plot & spline
        plt.scatter(store_data["Date"], store_data["Weekly_Sales"], color='darkgray', s=20)

        spline = UnivariateSpline(x, y, k=3, s=1e10)
        xs = np.linspace(x.min(), x.max(), 500)
        ys = spline(xs)

        xs_dates = [pd.Timestamp.fromordinal(int(d)) for d in xs]
        plt.plot(np.array(xs_dates), ys, color='orange', label='B-spline')
            
    axis = plt.gca()

    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=2))          
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))  # format as MMM-YYYY
    plt.xticks(rotation=45) 

    def millions(x, pos):
        return f'{x/1_000_000:.3f}'

    y_max = store_data["Weekly_Sales"].max()
    y_min = store_data["Weekly_Sales"].min()
    padding = .03 * y_max
    y_min_padding = y_min - padding
    y_max_padding = y_max + padding
    interval = (y_max - y_min) // 30
    plt.ylim(y_min_padding, y_max_padding)
    axis.yaxis.set_major_formatter(FuncFormatter(millions))
    axis.yaxis.set_major_locator(MultipleLocator(interval))

    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()

    plt.close()

def two_lines_plot(dataset: Dataset, col_one: str, col_two: str, output_file=None):
    plt.figure(figsize=(20,8))

    if not(col_one in dataset.columns and col_two in dataset.columns):
        raise KeyError("A column inserted is not apart of dataset")
    
    plt.plot(dataset.index, dataset[col_one], color = "deeppink", label = col_one)
    plt.plot(dataset.index, dataset[col_two], color = "green", label = col_two)
    plt.legend()

    column_one = re.sub(r"_", " ", col_one)
    column_two = re.sub(r"_", " ", col_two)
    plt.title(f"{column_one} & {column_two} over time (averaged across each month and store)")

    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()

    plt.close()

def correlation_heatmap(dataset: Dataset, output_file = None):
    corr = dataset.data.corr()
    plt.figure(figsize=(10, 10))
    sb.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
    )
    plt.title(f"Pearson Correlation Matrix")

    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()

    plt.close()

