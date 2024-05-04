#!/usr/bin/env python3.8
import pandas as pd
import numpy as np

import matplotlib as plt
import seaborn as sns



filename = 'netflix_titles.csv'
df =pd.read_csv(filename)

# print("Missing values per column:")
# print(missing_values)

def impute_missing_values(df):
    df['description'].fillna('', inplace=True)
    df['director'].fillna('Unknown', inplace=True)
    df['cast'].fillna('Unknown', inplace=True)
    df['rating'].fillna('Unknown', inplace=True)
    df['duration'] = df['duration'].str.extract('(\d+)').astype(float)
    median_duration = df['duration'].median()
    df['duration'].fillna(median_duration, inplace=True)
    missing_date_added = df[df['date_added'].isnull()]
    mode_date_added = df['date_added'].mode()[0]
    df['date_added'].fillna(mode_date_added, inplace=True)
    mode_country = df['country'].mode()[0]
    df['country'].fillna(mode_country, inplace=True)
    return df

# movie_duplicate = df.duplicated().sum()
# print(movie_duplicate)


# genre_data = df.iloc[0].Genre
# print(genre_data)

df.to_csv(filename, index=False)

# Alternatively, write to a new file
new_filename = 'amended_netflix_titles.csv'
df.to_csv(new_filename, index=False)

correlation= df.corr()

print(correlation)


