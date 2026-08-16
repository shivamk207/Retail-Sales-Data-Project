import numpy as np
import pandas as pd

np.random.seed(42)
n_rows = 1000

dates = pd.date_range(start="2024-01-01", end="2026-06-30", periods=n_rows)
categories = ["Technology", "Furniture", "Office Supplies"]
sub_categories = {
    "Technology": ["Phones", "Laptops", "Accessories"],
    "Furniture": ["Chairs", "Tables", "Bookcases"],
    "Office Supplies": ["Paper", "Binders", "Art"],
}
regions = ["North", "South", "East", "West"]

data = []
for i in range(n_rows):
    cat = np.random.choice(categories)
    sub_cat = np.random.choice(sub_categories[cat])
    region = np.random.choice(regions)
    quantity = np.random.randint(1, 10)
    base_price = np.random.uniform(20, 500)
    discount = round(np.random.choice([0.0, 0.1, 0.2, 0.3, 0.5]), 2)
    sales = round(quantity * base_price * (1 - discount), 2)
    profit = round(sales * np.random.uniform(0.05, 0.35) - (discount * 20), 2)

    data.append(
        [
            f"ORD-{10000+i}",
            dates[i],
            cat,
            sub_cat,
            region,
            sales,
            quantity,
            discount,
            profit,
        ]
    )

df = pd.DataFrame(
    data,
    columns=[
        "Order_ID",
        "Order_Date",
        "Category",
        "Sub_Category",
        "Region",
        "Sales",
        "Quantity",
        "Discount",
        "Profit",
    ],
)

# Introduce minor missing values for cleaning demonstration
df.loc[df.sample(frac=0.02).index, "Sales"] = np.nan
df.to_csv("sales_data.csv", index=False)
print("Dataset successfully generated: sales_data.csv")
