"""
Autoencoder Example (Unsupervised Neural Network)

Description:
Autoencoder compresses and reconstructs unlabelled animal-like data (height, weight).
Shows how well it learns to represent and reconstruct simple unlabelled data.

References:
- [Hinton, G. E., & Salakhutdinov, R. R. (2006). "Reducing the dimensionality of data with neural networks". Science, 313(5786), 504–507.](https://www.science.org/doi/10.1126/science.1127647)
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler

# Unlabelled animal-like data: [height (cm), weight (kg)]
X = np.array([
    [25, 4],    # Object 1
    [60, 20],   # Object 2
    [40, 2],    # Object 3
    [250, 5400],# Object 4
    [160, 400], # Object 5
])

# Normalize features for neural net
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Autoencoder: 2->1->2
input_layer = Input(shape=(2,))
encoded = Dense(1, activation='relu')(input_layer)
decoded = Dense(2, activation='sigmoid')(encoded)
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(X_scaled, X_scaled, epochs=300, verbose=0)

X_decoded = autoencoder.predict(X_scaled)
X_decoded_orig = scaler.inverse_transform(X_decoded)

plt.scatter(X[:, 0], X[:, 1], c='gray', s=100, label='Original')
plt.scatter(X_decoded_orig[:, 0], X_decoded_orig[:, 1], c='red', s=100, marker='x', label='Reconstructed')
for i in range(len(X)):
    plt.text(X[i, 0]+2, X[i, 1], f'Obj {i+1}', color='gray', fontsize=9)
    plt.text(X_decoded_orig[i, 0]+2, X_decoded_orig[i, 1], f'Obj {i+1}', color='red', fontsize=9)
plt.title("Autoencoder: Unlabelled Objects Original vs Reconstructed")
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.legend()
plt.show()

print("Note: The autoencoder works only with feature values, without any labels.")