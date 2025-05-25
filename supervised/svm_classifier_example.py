"""
Support Vector Machine (SVM) Classifier Example with Per-Feature Plots and Model Usage

Description:
This script demonstrates a Support Vector Machine Classifier for binary purchase prediction.
It uses a small sample dataset and:
- Plots each feature against the target (purchased) for visualization.
- Trains and evaluates an SVM classifier.
- Saves the trained model.
- Loads the saved model and uses it for predictions on new data.

Model Description:
Support Vector Machine (SVM) finds the optimal hyperplane that separates classes in the feature space.
It works well for small- to medium-sized datasets with clear class separation, and can handle non-linear boundaries with kernels.
SVM is robust to overfitting, especially in high-dimensional spaces.

References:
- [Cortes, C., & Vapnik, V. (1995). "Support-vector networks". Machine Learning, 20, 273–297.](https://link.springer.com/article/10.1007/BF00994018)
- [Scikit-learn SVC Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

This is a practical template for tabular classification: visualization, training, evaluation, saving, and inference.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

data = {
    "age": [22, 25, 47, 52, 46, 56, 55, 60, 35, 40],
    "salary": [25000, 32000, 47000, 52000, 46000, 56000, 55000, 60000, 35000, 40000],
    "purchased": [0, 0, 1, 1, 1, 1, 1, 1, 0, 1]
}
df = pd.DataFrame(data)
feature_columns = ["age", "salary"]
target = "purchased"
print("Sample data:")
print(df.to_string(index=False))

# Plot each feature vs. target
for feat in feature_columns:
    plt.figure(figsize=(6, 4))
    plt.scatter(df[feat], df[target], c=df[target].map({0:'red',1:'green'}))
    plt.xlabel(feat)
    plt.ylabel('Purchased')
    plt.title(f'{feat} vs Purchased')
    plt.yticks([0,1], ['No','Yes'])
    plt.grid(True)
    plt.show()

X = df[feature_columns]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model = SVC(kernel='linear', random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("SVM Classifier Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(model, "svm_model.pkl")
print("Model saved as svm_model.pkl")

# Load and use saved model
print("\n--- Using the saved model ---")
loaded_model = joblib.load("svm_model.pkl")
new_customers = pd.DataFrame({
    "age": [30, 58],
    "salary": [38000, 59000]
})
predictions = loaded_model.predict(new_customers)
for i, pred in enumerate(predictions):
    status = "Purchased" if pred == 1 else "Not Purchased"
    print(f"Customer {i+1}: {status} (features: {new_customers.iloc[i].to_dict()})")