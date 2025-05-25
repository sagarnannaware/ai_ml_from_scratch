"""
TensorFlow & Keras — Deep Learning

Description:
TensorFlow is a powerful open-source platform for machine learning and deep learning. Keras is its high-level API, making it easy to build and train neural networks.

Key Functionalities:
- Building and training neural networks (CNNs, RNNs, etc.)
- GPU acceleration
- Model serialization/export
- Handling large datasets
- Deployment to cloud, edge, or mobile

Sample Code:
"""

import tensorflow as tf
from tensorflow import keras

# Simple feedforward neural network
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(4,)),
    keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Keras model summary:")
model.summary()

# Example training:
# model.fit(train_X, train_y, epochs=10)