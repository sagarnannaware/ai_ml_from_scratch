"""
Pandas — Data Manipulation and Analysis

Description:
Pandas is the go-to library for data manipulation and analysis. It provides powerful data structures (Series and DataFrame) for handling structured data and a rich set of functions for cleaning, transforming, and analyzing data.

Key Functionalities:
- DataFrame and Series data structures
- Data filtering, aggregation, and grouping
- Handling missing data
- Merging, joining, reshaping datasets
- Reading/writing data from/to CSV, Excel, and SQL

Sample Code:
"""

import pandas as pd

# Create a DataFrame
data = {'name': ['Alice', 'Bob', 'Charlie'], 'score': [90, 85, 88]}
df = pd.DataFrame(data)

# Filter rows
high_scores = df[df['score'] > 85]

# Compute statistics
mean_score = df['score'].mean()

print("DataFrame:\n", df)
print("\nScores > 85:\n", high_scores)
print("\nMean score:", mean_score)