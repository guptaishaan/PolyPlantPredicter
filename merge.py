import pandas as pd
import numpy as np

# File paths
file1 = 'soil.csv'
file2 = 'ppt.csv'
file3 = 'yield.csv'

# Read the second column from each file
column1 = pd.read_csv(file1, usecols=[1], squeeze=True)

column2 = pd.read_csv(file2)
column2 = column2[column2['year'] == 2011]
column2 = column2.iloc[:, 1]  # Select the second column after filtering

column3 = pd.read_csv(file3)
column3 = column3[column3['year'] == 2011]
column3 = column3.iloc[:, 1]  # Select the second column after filtering



# Merge the columns into a single DataFrame
merged_data = pd.concat([column1, column2, column3], axis=1)

merged_data.to_csv('merged_dataV3.csv', index=False)
