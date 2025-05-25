"""
NumPy — Numerical Computing in Python

Description:
NumPy is the foundational package for numerical and scientific computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays efficiently.

Key Functionalities:
- Creation and manipulation of ndarrays (n-dimensional arrays)
- Broadcasting and vectorized operations
- Linear algebra (dot product, matrix multiplication)
- Random number generation
- Fast mathematical operations on entire arrays

Sample Code:
"""

import numpy as np

# Create arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Element-wise addition
sum_ab = a + b

# Dot product
dot_product = np.dot(a, b)

# 2D arrays and matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
matmul = np.matmul(A, B)

print("Array A:", a)
print("Array B:", b)
print("Sum:", sum_ab)
print("Dot product:", dot_product)
print("Matrix multiplication:\n", matmul)