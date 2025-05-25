"""
Scikit-learn — Classical Machine Learning

Description:
Scikit-learn is the most popular library for classical machine learning in Python. It provides simple and efficient tools for data mining and data analysis, built on top of NumPy and SciPy.

Key Functionalities:
- Algorithms for classification, regression, clustering, and dimensionality reduction
- Model selection and hyperparameter tuning
- Data preprocessing: scaling, encoding, imputation
- Pipelines for reproducible workflows

Sample Code:
"""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
clf = RandomForestClassifier()
clf.fit(X_train_scaled, y_train)

# Evaluate
accuracy = clf.score(X_test_scaled, y_test)
print("Random Forest accuracy on Iris dataset:", accuracy)