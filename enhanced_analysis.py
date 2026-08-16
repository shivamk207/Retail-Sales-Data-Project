import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

os.makedirs("reports/figures", exist_ok=True)

# ---------------------------------------------------------
# 1. DATA PREPARATION
# ---------------------------------------------------------
df = pd.read_csv("sales_data.csv")
df["Sales"] = df["Sales"].fillna(df["Sales"].median())

# Force dummy columns to be numeric floats
encoded_df = pd.get_dummies(
    df,
    columns=["Category", "Sub_Category", "Region"],
    drop_first=True,
    dtype=float
)

ignore_cols = ["Order_ID", "Order_Date", "Sales"]
features = [c for c in encoded_df.columns if c not in ignore_cols]

# Cast entire feature matrix to float64 to satisfy SHAP C-extensions
X = encoded_df[features].astype(float)
y = encoded_df["Sales"].astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 2. MODEL TRAINING & BENCHMARKING
# ---------------------------------------------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

xgb = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_r2 = r2_score(y_test, xgb_pred)

print("--- Model Comparison ---")
print(f"RF  RMSE: ${rf_rmse:.2f} | R2: {rf_r2:.4f}")
print(f"XGB RMSE: ${xgb_rmse:.2f} | R2: {xgb_r2:.4f}")

# ---------------------------------------------------------
# 3. SHAP EXPLAINABILITY ANALYSIS
# ---------------------------------------------------------
# Direct TreeExplainer handles XGBoost trees natively
explainer = shap.TreeExplainer(xgb)
shap_values = explainer(X_test)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()

output_path = "reports/figures/shap_summary.png"
plt.savefig(output_path)
plt.close()

print(f"SHAP summary plot saved to {output_path}")
