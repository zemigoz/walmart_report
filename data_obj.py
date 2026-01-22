import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
import kagglehub

from pathlib import Path
from sklearn.preprocessing import StandardScaler

class Dataset:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    # New functions
    def rarity_distribution(self, column: str):
        # return self.data["rarity"].value_counts().to_dict()
        # print(type(yugioh_data['rarity'].unique()))
        self._check_column(column)
        return self.data[column].value_counts()
    
    def to_datetime(self, column: str, dayfirst=False):
        self._check_column(column)
        self.data[column] = pd.to_datetime(self.data[column], dayfirst=dayfirst)
    
    def normalize(self, dummy = False):
        temp_data = self.data.copy()
        dates = temp_data["Date"]
        stores = temp_data["Store"]
        col = temp_data.drop(["Date", "Store"],axis=1).columns
        # for i in col:
        temp_data[col] = StandardScaler().fit_transform(temp_data[col])

        temp_data["Date"] = dates
        temp_data["Store"] = stores
        if dummy:
            temp_data = pd.get_dummies(temp_data, columns=["Store"], prefix="Store")
        # self.data = temp_data
        return temp_data

    def _check_column(self, column):
        if column not in self.columns:
            raise IndexError("Inputted string is not a column in dataset")
        
    # feature engineering hell
    def model_pipeline(self, smoothing_factor = 10, scale = True): 
        data = self.data
        data = data.sort_values(by=['Store', 'Date'])

        data["Log_Weekly_Sales"] = np.log1p(data["Weekly_Sales"])
        data["Date_Continuous"] = data['Date'].dt.year + (data['Date'].dt.month - 1) / 12
        data["month"] = data['Date'].dt.month

        # random stat class knowledge legendary pull rite here
        data["Cos_Month"] = np.cos(2 * np.pi * (data["month"] / 12))
        data["Sin_Month"] = np.sin(2 * np.pi * (data['month'] / 12))

        # Median over mean bc outliers (seasons)
        data['Sales_Lag_One'] = data.groupby('Store')['Log_Weekly_Sales'].shift(1)
        data['Sales_Lag_Two'] = data.groupby('Store')['Log_Weekly_Sales'].shift(2)
        data['Sales_Lag_One'] = data['Sales_Lag_One'].fillna(data['Log_Weekly_Sales'].median())
        data['Sales_Lag_Two'] = data['Sales_Lag_Two'].fillna(data['Log_Weekly_Sales'].median())
        
        #correlation between fuel and date is fine, this is a temporal series

        # 45 dummy variables is insane and will likely overfit. Best method for all models is to translate stores into a continuous variable
        # Encode using average up until term. Must be up to term and not global mean to avoid leakage since its temporal
        cumulative_sum = data.groupby('Store')['Log_Weekly_Sales'].cumsum().shift(1)
        cumulative_count = data.groupby('Store').cumcount()

        global_mean = data['Log_Weekly_Sales'].mean() # First entry is global
        data['Store_Sales_Encode'] = ((cumulative_sum + smoothing_factor * global_mean) / (cumulative_count + smoothing_factor)).fillna(global_mean) 

        data["Peak_Season"] = data['month'].apply(lambda x: 1 if x in [10, 11, 12] else 0)
        
        data = data.drop(columns = ["Date", 'month', "Weekly_Sales", "Store"])

        if scale:
            to_scale = [
                # "Log_Weekly_Sales",
                'Temperature', 
                'Fuel_Price', 
                'CPI', 
                'Unemployment',
                'Date_Continuous', 
                'Sales_Lag_One', 
                'Sales_Lag_Two',
                'Store_Sales_Encode'
            ]

            data[to_scale] = StandardScaler().fit_transform(data[to_scale])

        return Dataset(data=data)


    # Wrapper functions
    def info(self):
        return self.data.info()   
    
    @property
    def columns(self):
        return self.data.columns
    
    def head(self, n=5):
        return self.data.head(n)

    def describe(self):
        return self.data.describe()
    
    def nlargest(self, num: int, column: str):
        self._check_column(column)
        return self.data.nlargest(num, column)

    def nsmallest(self, num: int, column: str):
        self._check_column(column)
        return self.data.nsmallest(num, column)

    def isnull(self):
        return self.data.isnull()

    @property
    def shape(self):
        return self.data.shape

    def dropna(self):
        self.data = self.data.dropna()

    @property
    def index(self):
        return self.data.index

    # Hidden wrapper functions
    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return repr(self.data)