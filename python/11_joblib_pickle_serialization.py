"""
Joblib & Pickle — Model Serialization

Description:
Joblib and pickle are libraries for serializing (saving) and deserializing (loading) Python objects, which is essential for saving trained models to disk.

Key Functionalities:
- joblib: Efficient for large numpy arrays and scikit-learn models
- pickle: General-purpose Python object serialization
- Easy saving and loading of trained models

Sample Code:
"""

import joblib
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
# model.fit(X, y)  # Assume X, y are defined and model is trained

# Save model
joblib.dump(model, 'rf_model.pkl')

# Load model
loaded_model = joblib.load('rf_model.pkl')
print("Model saved and loaded with joblib.")