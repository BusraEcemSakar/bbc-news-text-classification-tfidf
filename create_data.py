import pandas as pd

# Load the dataset
data = pd.read_csv('path_to_bbc_dataset.csv')

# Inspect the data
print(data.head())
print(data['category'].value_counts())
