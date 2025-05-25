"""
K-Nearest Neighbors (KNN) Classifier Example with Per-Feature Plots and Model Usage

Description:
This script demonstrates a KNN Classifier for fruit type classification (apple/orange).
It uses a small sample dataset and:
- Plots each feature against the target (label) for visualization.
- Trains and evaluates a KNN classifier.
- Saves the trained model.
- Loads the saved model and uses it for predictions on new data.

Model Description:
K-Nearest Neighbors (KNN) is a simple, instance-based learning algorithm for classification and regression.
To classify a new data point, it finds the k closest points in the training set and assigns the most common class among them.
KNN is non-parametric, easy to understand, but can be slow for large datasets and sensitive to feature scaling.

References:
- [Cover, T., & Hart, P. (1967). "Nearest neighbor pattern classification". IEEE Transactions on Information Theory, 13(1), 21–27.](https://ieeexplore.ieee.org/document/1053964)
- [Scikit-learn KNeighborsClassifier Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

This is a practical template for tabular classification: visualization, training, evaluation, saving, and inference.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

data = {
    "weight": [150, 170, 140, 130, 180, 160, 120, 185, 175, 155],
    "color_score": [0.85, 0.80, 0.90, 0.88, 0.70, 0.65, 0.92, 0.72, 0.75, 0.80],
    "label": [1, 1, 1, 1, 0, 0, 1, 0, 0, 1]  # 1: Apple, 0: Orange
}
df = pd.DataFrame(data)
feature_columns = ["weight", "color_score"]
target = "label"
print("Sample data:")
print(df.to_string(index=False))

# Plot each feature vs. target
for feat in feature_columns:
    plt.figure(figsize=(6, 4))
    plt.scatter(df[feat], df[target], c=df[target].map({0:'orange',1:'green'}))
    plt.xlabel(feat)
    plt.ylabel('Fruit Type')
    plt.title(f'{feat} vs Fruit Type')
    plt.yticks([0,1], ['Orange','Apple'])
    plt.grid(True)
    plt.show()

X = df[feature_columns]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("KNN Classifier Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(model, "knn_model.pkl")
print("Model saved as knn_model.pkl")

# Load and use saved model
print("\n--- Using the saved model ---")
loaded_model = joblib.load("knn_model.pkl")
new_fruits = pd.DataFrame({
    "weight": [145, 180],
    "color_score": [0.91, 0.74]
})
predictions = loaded_model.predict(new_fruits)
for i, pred in enumerate(predictions):
    status = "Apple" if pred == 1 else "Orange"
    print(f"Fruit {i+1}: {status} (features: {new_fruits.iloc[i].to_dict()})")