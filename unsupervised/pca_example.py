"""
Principal Component Analysis (PCA) Example

Description:
This script demonstrates Principal Component Analysis (PCA), an unsupervised learning algorithm for dimensionality reduction.
It uses a small synthetic, unlabelled dataset (animal-like objects) and:
- Visualizes the original data in 2D.
- Applies PCA to reduce dimensions from 2D to 1D.
- Visualizes the reduced 1D data.

Note: The data is unlabelled; PCA finds patterns in the data without knowing animal types.

References:
- [Pearson, K. (1901). "On Lines and Planes of Closest Fit to Systems of Points in Space".](https://www.jstor.org/stable/2331913)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Unlabelled animal-like data: [height (cm), weight (kg)]
# Types are not provided to the model
X = np.array([
    [25, 4],    # Animal 1
    [60, 20],   # Animal 2
    [40, 2],    # Animal 3
    [250, 5400],# Animal 4
    [160, 400], # Animal 5
])

plt.scatter(X[:, 0], X[:, 1], color='green', s=80)
for i, (h, w) in enumerate(X):
    plt.text(h + 2, w, f'Animal {i+1}', fontsize=9)
plt.title("Unlabelled Animals: Height vs Weight")
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.show()

# Reduce to 1D for visualization
pca = PCA(n_components=1)
X_reduced = pca.fit_transform(X)
print("Original shape:", X.shape)
print("Reduced shape:", X_reduced.shape)
print("Reduced data:", X_reduced.ravel())

plt.scatter(X_reduced, np.zeros_like(X_reduced), color='blue', s=80)
for i, val in enumerate(X_reduced):
    plt.text(val + 5, 0, f'Animal {i+1}', fontsize=9)
plt.title("Animals after PCA (1D, Unlabelled)")
plt.xlabel("Principal Component 1")
plt.yticks([])
plt.show()

print("PCA finds patterns in the unlabelled data, projecting them onto the main axis of variation.")