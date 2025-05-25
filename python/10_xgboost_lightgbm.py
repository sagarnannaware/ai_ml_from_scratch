"""
XGBoost & LightGBM — Gradient Boosted Trees

Description:
XGBoost and LightGBM are high-performance libraries for gradient boosting, which is a powerful technique for structured/tabular data.

Key Functionalities:
- Fast, scalable training for classification and regression
- Handling of missing values and categorical features
- Feature importance visualization
- Model export for deployment

Sample Code:
"""

# XGBoost Example
import xgboost as xgb
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X, y)
print("XGBoost trained on Iris.")

# LightGBM Example
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X, y)
print("LightGBM trained on Iris.")