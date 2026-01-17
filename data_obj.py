import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
import kagglehub

from pathlib import Path

class Dataset:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    # New functions
    def rarity_distribution(self, column: str):
        # return self.data["rarity"].value_counts().to_dict()
        # print(type(yugioh_data['rarity'].unique()))
        if column not in self.columns:
            raise IndexError("Inputted string is not a column in dataset")

        return self.data[column].value_counts()
    
    def to_datetime(self, column: str, dayfirst=False):
        if column not in self.columns:
            raise IndexError("Inputted string is not a column in dataset")

        self.data[column] = pd.to_datetime(self.data[column], dayfirst=dayfirst)

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

    @property
    def shape(self):
        return self.data.shape

    # Hidden wrapper functions
    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return repr(self.data)