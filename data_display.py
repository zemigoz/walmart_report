import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from scipy.interpolate import UnivariateSpline, BSpline
from matplotlib.ticker import FuncFormatter, MultipleLocator
from typing import TypedDict

from data_obj import *

class DisplayDict(TypedDict):
    num_total_plots: int
    num_cols: int
    num_rows: int
    num_per_plots: int
    start: int

def display_pie_chart(data: Dataset, column: str, threshold: float, title: str) -> None:
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
    plt.pie(threshold_dist, labels=threshold_dist.index.astype(str).tolist(), autopct='%1.1f%%', startangle=90)
    plt.title(title)
    plt.show()

def display_plot(
    data: Dataset, 
    store_num: int,
    holidays=False, 
    options=0
):
    # if extra == 1:
    #     rolling = False
    #     holidays = True
    # elif extra == 2:
    #     rolling = True
    #     holidays = False
        
    store_data = data[data["Store"] == (store_num)]
    store_data = store_data.sort_values("Date")
    x = store_data["Date"].map(pd.Timestamp.toordinal)
    y = store_data["Weekly_Sales"].values 

    if options == 3:
        store_data["Weekly_Sales_Smooth"] = store_data["Weekly_Sales"].rolling(window=5, center=True).mean()
        y = store_data["Weekly_Sales_Smooth"]

    if holidays:
        holiday_zero = store_data[store_data["Holiday_Flag"] == 0]
        holiday_one = store_data[store_data["Holiday_Flag"] == 1]
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
        plt.scatter(holiday_zero["Date"], holiday_zero["Weekly_Sales"], color='skyblue', s=20)
        plt.scatter(holiday_one["Date"], holiday_one["Weekly_Sales"], color='salmon', s=20)

    elif options == 3: # Rolled/Smooth scatter
        plt.scatter(store_data["Date"], store_data["Weekly_Sales_Smooth"], color='dimgray', s=20)
        plt.title("Weekly Sales (Smoothed) Over Time Scatter Plot")

    elif options == 5: # Regular plot & spline
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

    plt.show()
