import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the CSV file into a pandas DataFrame
data = pd.read_csv("ann_pc_change.csv", delimiter=',', index_col=0)

# Remove commas and convert to float
data = data.replace(',', '', regex=True).astype(float)

# Define colors for the heatmap
cmap = sns.diverging_palette(240, 10, n=10, as_cmap=True)

# Replace empty cells with NaN and convert all values to float
data = data.replace(r'^\s*$', np.nan, regex=True).astype(float)

# Fill missing values with light gray
data = data.fillna(0.5)

# Define block size
block_size = 20

# Get number of rows and columns in the data
num_rows, num_cols = data.shape

# Iterate over each block and create a heatmap
for row_start in range(0, num_rows, block_size):
    for col_start in range(0, num_cols, block_size):
        # Extract block of data
        block_data = data.iloc[row_start:row_start + block_size, col_start:col_start + block_size]

        # Create the heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(block_data, cmap=cmap, center=0, annot=True, fmt=".1f", linewidths=0.5, linecolor='grey')

        # Add title
        plt.title(f'Heatmap Block ({row_start + 1}-{row_start + block_size}, {col_start + 1}-{col_start + block_size})')

        # Show plot
        plt.show()
