# Capstone: Customer Category Propensity Modelling

Predicts which product categories each customer is likely to buy next, using e-commerce behavioural
event logs (page visits, searches, add/remove-from-cart, purchases). User-level features are built
from the logs (aggregate, monetary, temporal, per-category and diversity features). Several models
are then compared on a multi-label propensity task: LightGBM, XGBoost, Random Forest and a Keras
neural network. A top-5-categories-per-user alternative is also tested.

**Course:** Capstone Project, Deree – The American College of Greece (Spring 2025)

## Tech
Python · pandas · NumPy · scikit-learn · LightGBM · XGBoost · TensorFlow/Keras · iterative-stratification

## Contents
- `Capstone_Category_Propensity.ipynb`: **final version**: data loading, feature engineering (event, ratio,
  monetary, temporal and diversity features for the top 30 categories), modelling, hyperparameter tuning,
  per-label threshold optimisation and top-K category suggestions
- `Category_Propensity.ipynb`: earlier version of the pipeline (top 15 categories)

## Data
The data is the **Universal Behavioral Modeling Dataset © 2025 Synerise SA** (RecSys Challenge 2025),
licensed [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). It is several GB, so it is
**not included**. Download it from the RecSys Challenge 2025 / Synerise website and place these in the
repository root: `product_buy.parquet`, `add_to_cart.parquet`, `remove_from_cart.parquet`,
`page_visit.parquet`, `search_query.parquet`, `product_properties.parquet`, and the `input/` and `target/` folders.

## How to run
```bash
pip install pandas numpy pyarrow scikit-learn lightgbm xgboost tensorflow iterative-stratification joblib jupyter
jupyter notebook Capstone_Category_Propensity.ipynb
```
