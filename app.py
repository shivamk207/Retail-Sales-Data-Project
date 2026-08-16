import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

st.set_page_config(
    page_title="Retail Sales Dashboard",
    layout="wide"
)

st.title("Retail Sales & Analytics Platform")

# Sidebar Data Loading
uploaded_file = st.sidebar.file_uploader(
    "Upload Sales CSV", type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("sales_data.csv")

df["Order_Date"] = pd.to_datetime(df["Order_Date"])
df["Sales"] = df["Sales"].fillna(df["Sales"].median())

# Sidebar Filters
region = st.sidebar.multiselect(
    "Select Region",
    options=df["Region"].unique(),
    default=df["Region"].unique()
)
category = st.sidebar.multiselect(
    "Select Category",
    options=df["Category"].unique(),
    default=df["Category"].unique()
)

filtered_df = df[
    (df["Region"].isin(region)) & (df["Category"].isin(category))
]

# Key Performance Indicators
col1, col2, col3 = st.columns(3)
col1.metric("Total Sales", f"${filtered_df['Sales'].sum():,.2f}")
col2.metric("Total Profit", f"${filtered_df['Profit'].sum():,.2f}")
col3.metric("Total Orders", f"{len(filtered_df):,}")

# Interactive Charts
st.subheader("Sales Performance Analysis")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig_cat = px.bar(
        filtered_df,
        x="Category",
        y="Sales",
        color="Sub_Category",
        title="Sales by Category & Sub-Category",
        barmode="group"
    )
    st.plotly_chart(fig_cat, use_container_width=True)

with chart_col2:
    fig_discount = px.scatter(
        filtered_df,
        x="Discount",
        y="Profit",
        color="Category",
        title="Impact of Discount on Profit"
    )
    st.plotly_chart(fig_discount, use_container_width=True)

# Predictive Modeling Interface
st.subheader("Model Benchmarking (Random Forest vs XGBoost)")
encoded_df = pd.get_dummies(
    df,
    columns=["Category", "Sub_Category", "Region"],
    drop_first=True
)

features = [
    c for c in encoded_df.columns
    if c not in ["Order_ID", "Order_Date", "Sales"]
]
X = encoded_df[features]
y = encoded_df["Sales"]

rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
xgb_model = XGBRegressor(n_estimators=50, random_state=42)

rf_model.fit(X, y)
xgb_model.fit(X, y)

rf_r2 = r2_score(y, rf_model.predict(X))
xgb_r2 = r2_score(y, xgb_model.predict(X))

res_col1, res_col2 = st.columns(2)
res_col1.metric("Random Forest R² Score", f"{rf_r2:.4f}")
res_col2.metric("XGBoost R² Score", f"{xgb_r2:.4f}")
