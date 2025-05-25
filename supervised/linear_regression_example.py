"""
Linear Regression Example with Per-Feature Plots and Model Usage

Description:
This script demonstrates Linear Regression for predicting house prices from features.
It uses a small sample dataset and:
- Plots each feature against the target (price) for visualization.
- Trains and evaluates a Linear Regression model.
- Saves the trained model.
- Loads the saved model and uses it for predictions on new data.

Model Description:
Linear Regression is a simple algorithm for predicting continuous target values.
It models the relationship between one or more features and the target as a straight line (linear equation).
It is interpretable, fast, and works well when the relationship is approximately linear.

References:
- [Galton, F. (1886). "Regression towards mediocrity in hereditary stature". The Journal of the Anthropological Institute of Great Britain and Ireland, 15, 246–263.](https://www.jstor.org/stable/2841583)
- [Scikit-learn LinearRegression Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

This is a practical template for tabular regression: visualization, training, evaluation, saving, and inference.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

data = {
    "sqft": [1500, 1700, 1600, 1800, 2000, 2100, 1200, 1300, 1250, 1950],
    "bedrooms": [3, 4, 3, 4, 5, 4, 2, 3, 2, 4],
    "price": [200000, 250000, 220000, 270000, 315000, 325000, 150000, 160000, 155000, 290000]
}
df = pd.DataFrame(data)
feature_columns = ["sqft", "bedrooms"]
target = "price"
print("Sample data:")
print(df.to_string(index=False))

# Plot each feature vs. target (regression)
for feat in feature_columns:
    plt.figure(figsize=(6, 4))
    plt.scatter(df[feat], df[target], c='blue')
    plt.xlabel(feat)
    plt.ylabel('Price')
    plt.title(f'{feat} vs Price')
    plt.grid(True)
    plt.show()

X = df[feature_columns]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred))

joblib.dump(model, "linear_regression_model.pkl")
print("Model saved as linear_regression_model.pkl")

# Load and use saved model
print("\n--- Using the saved model ---")
loaded_model = joblib.load("linear_regression_model.pkl")
new_houses = pd.DataFrame({
    "sqft": [1550, 2050],
    "bedrooms": [3, 5]
})
predictions = loaded_model.predict(new_houses)
for i, pred in enumerate(predictions):
    print(f"House {i+1}: Predicted Price: {pred:.2f} (features: {new_houses.iloc[i].to_dict()})")