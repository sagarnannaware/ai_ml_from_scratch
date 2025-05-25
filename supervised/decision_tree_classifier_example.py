"""
Decision Tree Classifier Example with Per-Feature Plots and Model Usage

Description:
This script demonstrates a Decision Tree Classifier for loan approval prediction.
It uses a small sample dataset and:
- Plots each feature against the target (approved) for visualization.
- Trains and evaluates a Decision Tree classifier.
- Saves the trained model.
- Loads the saved model and uses it for predictions on new data.

Model Description:
A Decision Tree is a non-linear, tree-structured model for classification and regression.
It splits the data into branches using feature-based questions, forming a tree from root to leaf.
Decision Trees are easy to visualize and interpret, can model non-linear relationships, but may overfit on small datasets.

References:
- [Quinlan, J. R. (1986). "Induction of Decision Trees". Machine Learning, 1, 81–106.](https://link.springer.com/article/10.1023/A:1022643204877)
- [Scikit-learn Decision Tree Classifier Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

This is a practical template for tabular classification: visualization, training, evaluation, saving, and inference.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

data = {
    "credit_score": [600, 650, 700, 720, 680, 690, 710, 730, 740, 750],
    "income": [30000, 40000, 50000, 60000, 35000, 45000, 55000, 65000, 70000, 80000],
    "approved": [0, 0, 1, 1, 0, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)
feature_columns = ["credit_score", "income"]
target = "approved"
print("Sample data:")
print(df.to_string(index=False))

# Plot each feature vs. target
for feat in feature_columns:
    plt.figure(figsize=(6, 4))
    plt.scatter(df[feat], df[target], c=df[target].map({0:'red',1:'green'}))
    plt.xlabel(feat)
    plt.ylabel('Approved')
    plt.title(f'{feat} vs Approved')
    plt.yticks([0,1], ['No','Yes'])
    plt.grid(True)
    plt.show()

X = df[feature_columns]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = DecisionTreeClassifier(random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(model, "decision_tree_model.pkl")
print("Model saved as decision_tree_model.pkl")

# Load and use saved model
print("\n--- Using the saved model ---")
loaded_model = joblib.load("decision_tree_model.pkl")
new_loans = pd.DataFrame({
    "credit_score": [660, 720],
    "income": [37000, 70000]
})
predictions = loaded_model.predict(new_loans)
for i, pred in enumerate(predictions):
    status = "Approved" if pred == 1 else "Not Approved"
    print(f"Loan Applicant {i+1}: {status} (features: {new_loans.iloc[i].to_dict()})")