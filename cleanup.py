import pandas as pd
import numpy as np

# Read the CSV file
data = pd.read_csv('merged_data.csv')

# Replace missing values with the column averages
for column in data.columns:
    average_value = np.mean(data[column])
    data[column].fillna(average_value, inplace=True)

# Save the updated data to a new CSV file
data.to_csv('final_data.csv', index=False)
