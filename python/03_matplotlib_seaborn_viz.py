"""
Matplotlib & Seaborn — Data Visualization in Python

Description:
Matplotlib is the standard Python library for creating static, animated, and interactive visualizations. Seaborn is built on top of matplotlib and makes it easy to create attractive and informative statistical graphics.

Key Functionalities:
- Line, bar, scatter, histogram, and pie charts (Matplotlib)
- Statistical plots (Seaborn): heatmaps, pairplots, violin plots, etc.
- Customizable figures, axes, and themes
- Publication-quality visualizations

Sample Code:
"""

import matplotlib.pyplot as plt
import seaborn as sns

data = [1, 2, 2, 3, 3, 3, 4]

# Matplotlib histogram
plt.hist(data, bins=4)
plt.title("Matplotlib Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# Seaborn histogram
sns.histplot(data)
plt.title("Seaborn Histogram Example")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()