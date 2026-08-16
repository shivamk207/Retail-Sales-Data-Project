# Summary Findings: Retail Sales Analysis & Predictive Model

## Key Analytical Insights
* **Primary Revenue Drivers:** The **Technology** category generates the highest total sales volume, driven largely by high base unit pricing in laptops and phones.
* **Discount Sensitivity:** Discount rates above **30%** severely compress profit margins. Transactions with a 50% discount rate consistently yield negative or near-zero profits across all regions.
* **Regional Distribution:** Sales volumes remain evenly distributed across North, South, East, and West regions, though the East region exhibits slightly higher profitability per unit sold.

## Predictive Model Performance
* **Algorithm:** Random Forest Regressor
* **Target Variable:** Transaction Sales Value ($)
* **Model Accuracy:** Achieved an $R^2$ score demonstrating strong predictive correlation based on transaction features including item category, quantity, discount level, and region.

## Business Recommendations
1. **Cap Discounts:** Restrict maximum baseline promotional discounts to 20% to prevent margin erosion.
2. **Stock Allocation:** Prioritize inventory replenishment for Technology sub-categories ahead of high-volume sales periods.