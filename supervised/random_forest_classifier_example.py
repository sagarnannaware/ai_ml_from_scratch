"""
Random Forest Classifier Example with Per-Feature Plots and Model Usage

Description:
This script demonstrates a Random Forest Classifier for predicting house purchase from multiple features.
It uses a small sample dataset and:
- Plots each feature against the target (buy_house) for visualization.
- Trains and evaluates a Random Forest classifier.
- Shows feature importance.
- Saves the trained model.
- Loads the saved model and uses it for predictions on new data.

Model Description:
Random Forest is an ensemble machine learning method for classification and regression.
It builds multiple decision trees using random subsets of the data and features, and combines their outputs for robust predictions.
Random Forest reduces overfitting, improves accuracy, and provides feature importance scores.
It works well for tabular data, handles both categorical and numerical features, and is robust to noise and outliers.

References:
- [Breiman, L. (2001). "Random Forests". Machine Learning, 45(1), 5–32.](https://link.springer.com/article/10.1023/A:1010933404324)
- [Scikit-learn Random Forest Classifier Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

This is a practical template for tabular classification: visualization, training, evaluation, saving, and inference.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Data
data = {
    "age":              [25, 40, 31, 36, 52, 46, 29, 28, 35, 41],
    "income":           [35000, 85000, 57000, 62000, 120000, 110000, 45000, 40000, 95000, 99000],
    "married":          [0, 1, 1, 0, 1, 1, 0, 0, 1, 0],
    "savings":          [5000, 40000, 12000, 25000, 80000, 60000, 8000, 3000, 70000, 15000],
    "loan_history":     [0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
    "has_children":     [0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
    "employment_years": [2, 10, 7, 6, 20, 18, 4, 3, 15, 9],
    "buy_house":        [0, 1, 1, 0, 1, 1, 0, 0, 1, 1]
}
df = pd.DataFrame(data)
feature_columns = ["age", "income", "married", "savings", "loan_history", "has_children", "employment_years"]
target = "buy_house"
print("Sample data:")
print(df.to_string(index=False))

# Plot each feature vs. target
for feat in feature_columns:
    plt.figure(figsize=(6, 4))
    plt.scatter(df[feat], df[target], c=df[target].map({0:'red',1:'green'}))
    plt.xlabel(feat)
    plt.ylabel('Buy House')
    plt.title(f'{feat} vs Buy House')
    plt.yticks([0,1], ['No','Yes'])
    plt.grid(True)
    plt.show()

# Model Training
X = df[feature_columns]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))

# Feature Importance
importances = model.feature_importances_
plt.figure(figsize=(8,5))
plt.bar(feature_columns, importances, color='skyblue')
plt.title('Feature Importances (Random Forest)')
plt.ylabel('Importance')
plt.xlabel('Feature')
plt.show()

joblib.dump(model, "house_purchase_model.pkl")
print("Model saved as house_purchase_model.pkl")

# Load and use the saved model for new predictions
print("\n--- Using the saved model ---")
loaded_model = joblib.load("house_purchase_model.pkl")
new_customers = pd.DataFrame({
    "age": [32, 55],
    "income": [60000, 125000],
    "married": [1, 1],
    "savings": [20000, 90000],
    "loan_history": [0, 1],
    "has_children": [0, 1],
    "employment_years": [8, 22],
})
predictions = loaded_model.predict(new_customers)
for i, pred in enumerate(predictions):
    status = "Will Buy House" if pred == 1 else "Will Not Buy House"
    print(f"Customer {i+1}: {status} (features: {new_customers.iloc[i].to_dict()})")