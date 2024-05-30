import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

# Read the CSV file
filename = 'netflix_titles.csv'
df = pd.read_csv(filename)

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values per column:")
print(missing_values)

df = df[df['type'] == 'Movie']

# Define the function to impute missing values
def impute_missing_values(df):
    df['description'].fillna('', inplace=True)
    df['director'].fillna('Unknown', inplace=True)
    df['cast'].fillna('Unknown', inplace=True)
    df['rating'].fillna('Unknown', inplace=True)
    df['duration'] = df['duration'].astype(str).str.extract('(\d+)')
    df['duration'] = df['duration'].astype(float)
    
    # Handle missing and zero values
    mean_duration = df['duration'][df['duration'] != 0].mean()
    df['duration'].replace(0, np.nan, inplace=True)
    df['duration'].fillna(mean_duration, inplace=True)
    
    mode_date_added = df['date_added'].mode()[0]
    df['date_added'].fillna(mode_date_added, inplace=True)
    mode_country = df['country'].mode()[0]
    df['country'].fillna(mode_country, inplace=True)
    return df

# Impute missing values
df = impute_missing_values(df)

# Verify if the imputation worked correctly
# print("Missing values in 'duration' after imputation:", df['duration'].isnull().sum())
# print("Number of zero values in 'duration' after imputation:", (df['duration'] == 0).sum())

# Save the modified DataFrame to the original file and a new file
#ADD Theme column based on description

# df = df.drop(columns=['Genre'])
# df = df.rename(columns={'listed_in': 'genre'})

themes = [
    "Love", "Forgiveness", "Loss", "Hope", "Revenge", "Betrayal", "Courage",
    "Friendship", "Redemption", "Sacrifice", "Freedom", "Loyalty", "Greed",
    "Power", "Justice", "Survival", "Trust", "Isolation", "Ambition",
    "Deception", "Fear", "Family", "Identity", "Guilt", "Healing",
    "Ambiguity", "Conflict", "Transformation", "Honor", "Resilience",
    "Innocence", "Duty", "Destiny", "Ambivalence", "Mortality", "Despair",
    "Unity", "Tradition", "Faith", "Obsession", "Grief", "Rebellion",
    "Discovery", "Rivalry", "Compassion", "Mystery", "Joy", "Temptation",
    "Alienation", "Legacy", "Nostalgia", "Regret", "Ambivalence", "Courage",
    "Transformation", "Struggle", "Desperation", "Escape", "Reconciliation",
    "Adventure", "Redemption", "Manipulation", "Sacrifice", "Spirituality",
    "Passion", "Vengeance", "Prejudice", "Alienation", "Empowerment", "Fantasy",
    "Dilemma", "Melancholy", "Pursuit", "Reckoning", "Turmoil", "Forging",
    "Discovery", "Journey", "Rivalry", "Quest", "Isolation", "Resolve", "Vision",
    "Ambition", "Intrigue", "Prophecy", "Anarchy", "War", "Reflection",
    "Conquest", "Unity", "Euphoria", "Dream", "Horror", "Fate", "Disguise",
    "Escape", "Surrender", "Evolution","Magic","Death", "Corrupt", "Crime", "Evil", "Tragedy"
]

# Function to extract themes from the descriptionp
def extract_themes(description):
    found_themes = [theme for theme in themes if theme.lower() in description.lower()]
    return ', '.join(found_themes) if found_themes else 'None'

# Apply the function to the description column
df['theme'] = df['description'].apply(extract_themes)

# Verify the new column
# print(df[['description', 'theme']].head())

# Save the modified DataFrame to the original file and a new file
df.to_csv(filename, index=False)
new_filename = 'amended_netflix_titles.csv'
df.to_csv(new_filename, index=False)

scaler = StandardScaler()
df['duration'] = scaler.fit_transform(df[['duration']])

standardised_filename = 'standardised_netflix_titles.csv'
df.to_csv(standardised_filename, index=False)

# Extract relevant features for building user profiles and recommendations
features1 = df[['title','genre','country']]
features2= df[['title', 'rating', 'duration', 'theme']]             

# Print the first few rows of the features
print(features1.head())
print(features2.head())
# print("Duration column values:")
# print(df['duration'])
# # Data visualisations
# plt.figure(figsize=(10, 6))
# plt.hist(df['duration'], bins=30, edgecolor='k', alpha=0.7)
# plt.title('Distribution of Movie Durations')
# plt.xlabel('Duration (minutes)')
# plt.ylabel('Number of Movies')
# plt.grid(True)
# plt.show()
# theme_counts = Counter(theme for themes in df['theme'] for theme in themes.split(', ') if theme != 'None')

# # Create a DataFrame from the theme counts
# theme_counts_df = pd.DataFrame.from_dict(theme_counts, orient='index', columns=['count']).reset_index()
# theme_counts_df = theme_counts_df.rename(columns={'index': 'theme'}).sort_values(by='count', ascending=False)

# # # Plot the bar chart
# # plt.figure(figsize=(12, 8))
# # plt.bar(theme_counts_df['theme'], theme_counts_df['count'], color='skyblue')
# # plt.xlabel('Themes')
# # plt.ylabel('Number of Movies')
# # plt.title('Distribution of Themes in Movie Descriptions')
# # plt.xticks(rotation=90)
# # plt.grid(axis='y')
# # plt.show()




# #Features - content based filtering
# features = df[['title', 'genre', 'country', 'rating', 'duration', 'theme']]


# print(features.head())


# normalizer = MinMaxScaler()
# df['duration_normalised'] = normalizer.fit_transform(df[['duration']])

# normalised_filename = 'normalised_netflix_titles.csv'
# df.to_csv(normalised_filename, index=False)

# # Calculate correlation only on numerical columns
# numerical_df = df.select_dtypes(include=[np.number])
# correlation = numerical_df.corr()

# print(correlation)

# # Calculate correlation only on numerical columns
# numerical_df = df.select_dtypes(include=[np.number])
# correlation = numerical_df.corr()

# print(correlation)



