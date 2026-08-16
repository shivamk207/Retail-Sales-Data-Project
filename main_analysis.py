import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

os.makedirs("reports/figures", exist_ok=True)

# ---------------------------------------------------------
# 1. DATA LOADING & CLEANING
# ---------------------------------------------------------
df = pd.read_csv("sales_data.csv")

# Impute missing sales values with column median
df["Sales"] = df["Sales"].fillna(df["Sales"].median())

# Format date column and extract temporal features
df["Order_Date"] = pd.to_datetime(df["Order_Date"])
df["Year"] = df["Order_Date"].dt.year
df["Month"] = df["Order_Date"].dt.month
df["YearMonth"] = df["Order_Date"].dt.to_period("M")
df["DayOfWeek"] = df["Order_Date"].dt.dayofweek

# ---------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS & VISUALIZATION
# ---------------------------------------------------------
sns.set_theme(style="whitegrid")

# Plot 1: Monthly Sales Trend
plt.figure(figsize=(10, 5))
monthly_sales = df.groupby("YearMonth")["Sales"].sum().reset_index()
monthly_sales["YearMonth"] = monthly_sales["YearMonth"].astype(str)
sns.lineplot(
    data=monthly_sales, x="YearMonth", y="Sales", marker="o", color="#1f77b4"
)
plt.title("Monthly Sales Trend (2024 - 2026)", fontsize=14, fontweight="bold")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("reports/figures/monthly_sales_trend.png")
plt.close()

# Plot 2: Sales and Profit by Category
plt.figure(figsize=(8, 5))
cat_perf = (
    df.groupby("Category")[["Sales", "Profit"]]
    .sum()
    .reset_index()
    .melt(id_vars="Category")
)
sns.barplot(data=cat_perf, x="Category", y="value", hue="variable")
plt.title("Total Sales & Profit by Category", fontsize=14, fontweight="bold")
plt.ylabel("Amount ($)")
plt.tight_layout()
plt.savefig("reports/figures/category_performance.png")
plt.close()

# Plot 3: Discount vs Profit Impact
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Discount", y="Profit", palette="Set2")
plt.title(
    "Impact of Discount Rate on Profitability",
    fontsize=14,
    fontweight="bold"
        )
plt.tight_layout()
plt.savefig("reports/figures/discount_vs_profit.png")
plt.close()

# ---------------------------------------------------------
# 3. PREDICTIVE MODELING (Random Forest Regressor)
# ---------------------------------------------------------
# Encode categorical variables for modeling
encoded_df = pd.get_dummies(
    df, columns=["Category", "Sub_Category", "Region"], drop_first=True
)

features = [
    c
    for c in encoded_df.columns
    if c
    not in [
        "Order_ID",
        "Order_Date",
        "YearMonth",
        "Sales",
    ]
]
X = encoded_df[features]
y = encoded_df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("--- Model Evaluation Metrics ---")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
print(f"R-squared Score (R2): {r2:.4f}")
