"""
Logistic Regression Example with Per-Feature Plots and Model Usage

Description:
This script demonstrates Logistic Regression for binary classification (pass/fail).
It uses a small sample dataset and:
- Plots each feature against the target (passed) for visualization.
- Trains and evaluates a Logistic Regression model.
- Saves the trained model.
- Loads the saved model and uses it for predictions on new data.

Model Description:
Logistic Regression is a linear model for binary classification.
It estimates the probability that an input belongs to a class by applying the logistic (sigmoid) function to a linear combination of the features.
It is simple, interpretable, fast to train, and works well when the relationship between features and the target is roughly linear.

References:
- [Cox, D. R. (1958). "The Regression Analysis of Binary Sequences". Journal of the Royal Statistical Society. Series B (Methodological), 20(2), 215–242.](https://www.jstor.org/stable/2983890)
- [Scikit-learn Logistic Regression Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

This is a practical template for binary classification: visualization, training, evaluation, saving, and inference.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Data
data = {
    "hours_studied": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    "attendance": [75, 80, 85, 90, 95, 100, 85, 80, 95, 90],
    "passed": [0, 0, 0, 1, 1, 1, 1, 0, 1, 1]
}
df = pd.DataFrame(data)
feature_columns = ["hours_studied", "attendance"]
target = "passed"
print("Sample data:")
print(df.to_string(index=False))

# Plot each feature vs. target
for feat in feature_columns:
    plt.figure(figsize=(6, 4))
    plt.scatter(df[feat], df[target], c=df[target].map({0:'red',1:'green'}))
    plt.xlabel(feat)
    plt.ylabel('Passed')
    plt.title(f'{feat} vs Passed')
    plt.yticks([0,1], ['No','Yes'])
    plt.grid(True)
    plt.show()

X = df[feature_columns]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(model, "logistic_regression_model.pkl")
print("Model saved as logistic_regression_model.pkl")

# Load and use saved model
print("\n--- Using the saved model ---")
loaded_model = joblib.load("logistic_regression_model.pkl")
new_students = pd.DataFrame({
    "hours_studied": [5, 15],
    "attendance": [82, 99]
})
predictions = loaded_model.predict(new_students)
for i, pred in enumerate(predictions):
    status = "Passed" if pred == 1 else "Failed"
    print(f"Student {i+1}: {status} (features: {new_students.iloc[i].to_dict()})")