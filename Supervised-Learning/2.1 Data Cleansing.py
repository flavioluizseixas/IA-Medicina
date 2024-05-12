# Import required libraries
import pandas as pd
import numpy as np
from functions import *

df = pd.read_csv("data/csv/cleaning.csv")

# Convert date_of_sale from string to datetime
df["date_of_sale"] = pd.to_datetime(df["date_of_sale"], format='mixed', dayfirst=True)

# Find non-numerics in number_of_bedrooms
non_nums = df[~df["number_of_bedrooms"].str.isnumeric()]["number_of_bedrooms"].unique()
# Replace non-numerics in number_of_bedrooms with nulls
df["number_of_bedrooms"] = df["number_of_bedrooms"].replace(non_nums, np.nan)
# Convert number_of_bedrooms from string to numeric
df['number_of_bedrooms'] = pd.to_numeric(df['number_of_bedrooms'])

# Remove non-numeric characters from price
df["price"] = df["price"].apply(lambda x: x.replace('£', '') if type(x) is str else x)
df["price"] = df["price"].apply(lambda x: x.replace(',', '') if type(x) is str else x)
# Convert price from string to numeric
df['price'] = pd.to_numeric(df['price'])
# Replace 0s with nulls in price
df["price"] = df["price"].replace([0], np.nan)

# Replace misspelling
df["type"] = df["type"].replace(['teraced'], 'terraced')


outliers_mask = find_outliers(df.number_of_bedrooms)
df.loc[outliers_mask,'number_of_bedrooms'] = np.nan

mean = round(df["price"].median())     # calculate the mean for the column
df.loc[df["price"].isnull(), "price"] = mean
df.loc[:, 'year'] = df['date_of_sale'].dt.year
df.loc[:, 'price'] = df.groupby(['location', 'number_of_bedrooms', 'year'])['price'].transform(lambda x: x.fillna(x.mean()))
