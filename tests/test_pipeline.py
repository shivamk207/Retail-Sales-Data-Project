import numpy as np
import pandas as pd


def test_data_cleaning():
    data = {
        "Sales": [100.0, np.nan, 300.0],
        "Quantity": [1, 2, 3],
        "Discount": [0.1, 0.2, 0.0],
    }
    df = pd.DataFrame(data)

    # Perform cleaning
    df["Sales"] = df["Sales"].fillna(df["Sales"].median())

    assert df["Sales"].isnull().sum() == 0
    assert df.loc[1, "Sales"] == 200.0


def test_feature_encoding():
    data = {
        "Category": ["Technology", "Furniture"],
        "Sales": [500.0, 200.0],
    }
    df = pd.DataFrame(data)
    encoded = pd.get_dummies(df, columns=["Category"])

    assert "Category_Technology" in encoded.columns
    assert "Category_Furniture" in encoded.columns
