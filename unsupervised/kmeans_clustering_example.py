"""
K-Means Clustering Example

Description:
This script demonstrates K-Means clustering, an unsupervised learning algorithm, to group data points based on their similarity.
It uses a small, simple dataset with unlabelled fruit-like objects and:
- Visualizes the data points.
- Applies K-Means to find clusters.
- Visualizes the resulting clusters and centroids.

Note: The data is unlabelled (no fruit names given to the model), so K-Means groups by feature similarity, not by true type.

References:
- [MacQueen, J. (1967). "Some Methods for Classification and Analysis of Multivariate Observations".](https://projecteuclid.org/euclid.bsmsp/1200512992)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Unlabelled data: [weight (grams), color_score (0-1)]
# These could represent apples and oranges, but the model does not know this!
X = np.array([
    [150, 0.82],  # Fruit 1
    [170, 0.85],  # Fruit 2
    [140, 0.80],  # Fruit 3
    [130, 0.55],  # Fruit 4
    [120, 0.58],  # Fruit 5
    [110, 0.60],  # Fruit 6
])

plt.scatter(X[:, 0], X[:, 1], color='gray', s=100)
for i, (w, c) in enumerate(X):
    plt.text(w + 2, c, f'Fruit {i+1}', fontsize=9)
plt.title("Unlabelled Fruits: Weight vs Color Score")
plt.xlabel("Weight (grams)")
plt.ylabel("Color Score")
plt.show()

kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='cool', s=100, label='Clustered Fruits')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, marker='X', label='Centroids')
plt.title("K-Means Clustering: Unlabelled Fruits")
plt.xlabel("Weight (grams)")
plt.ylabel("Color Score")
plt.legend()
plt.show()

print("Cluster assignments (0/1, arbitrary labels):", labels)
print("Cluster centers:\n", centers)
print("\nNotice that the model forms clusters based solely on feature similarity, not fruit names.")