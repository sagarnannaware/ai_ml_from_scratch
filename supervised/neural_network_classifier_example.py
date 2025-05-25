"""
Neural Network (MLP) Classifier Example with Per-Feature Plots and Model Usage

Description:
This script demonstrates a Multi-Layer Perceptron (MLP) classifier for customer churn prediction.
It uses a small sample dataset and:
- Plots each feature against the target (churn) for visualization.
- Trains and evaluates a neural network classifier.
- Saves the trained model.
- Loads the saved model and uses it for predictions on new data.

Model Description:
A Multi-Layer Perceptron (MLP) is a type of feedforward artificial neural network.
It consists of input, hidden, and output layers of neurons, and can learn complex non-linear relationships.
MLPs are flexible and powerful, but may require more data and tuning than simpler models.

References:
- [Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). "Learning representations by back-propagating errors". Nature, 323(6088), 533–536.](https://www.nature.com/articles/323533a0)
- [Scikit-learn MLPClassifier Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)

This is a practical template for tabular classification: visualization, training, evaluation, saving, and inference.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

data = {
    "monthly_minutes": [200, 250, 300, 150, 400, 100, 350, 220, 170, 390],
    "customer_service_calls": [1, 0, 2, 5, 0, 6, 1, 2, 4, 0],
    "international_plan": [0, 0, 1, 0, 1, 1, 0, 1, 0, 1],
    "churn": [0, 0, 1, 1, 0, 1, 0, 1, 1, 0]
}
df = pd.DataFrame(data)
feature_columns = ["monthly_minutes", "customer_service_calls", "international_plan"]
target = "churn"
print("Sample data:")
print(df.to_string(index=False))

# Plot each feature vs. target
for feat in feature_columns:
    plt.figure(figsize=(6, 4))
    plt.scatter(df[feat], df[target], c=df[target].map({0:'blue',1:'orange'}))
    plt.xlabel(feat)
    plt.ylabel('Churn')
    plt.title(f'{feat} vs Churn')
    plt.yticks([0,1], ['Stayed','Churned'])
    plt.grid(True)
    plt.show()

X = df[feature_columns]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Neural Network Classifier Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(model, "mlp_classifier_model.pkl")
print("Model saved as mlp_classifier_model.pkl")

# Load and use saved model
print("\n--- Using the saved model ---")
loaded_model = joblib.load("mlp_classifier_model.pkl")
new_customers = pd.DataFrame({
    "monthly_minutes": [210, 390],
    "customer_service_calls": [0, 3],
    "international_plan": [0, 1]
})
predictions = loaded_model.predict(new_customers)
for i, pred in enumerate(predictions):
    status = "Churned" if pred == 1 else "Stayed"
    print(f"Customer {i+1}: {status} (features: {new_customers.iloc[i].to_dict()})")