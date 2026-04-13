# California Housing Regression

This project compares several regression models on the California Housing dataset.

## Goal

The goal of this project is to predict median house values in California districts using machine learning regression models.

## Dataset

The project uses the California Housing dataset available in scikit-learn.

Features include:
- median income
- house age
- average number of rooms
- average number of bedrooms
- population
- households
- latitude
- longitude

Target:
- median house value

## Models used

The following models were tested:
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

## Train / Test Split Results

Results from a single train/test split:

- **Linear Regression** — R²: **0.6401**, RMSE: **70156.12**
- **Random Forest Regressor** — R²: **0.8239**, RMSE: **49075.70**
- **Gradient Boosting Regressor** — R²: **0.7742**, RMSE: **55564.37**

## Cross-Validation Results

5-fold cross-validation was performed using:

`KFold(n_splits=5, shuffle=True, random_state=42)`

Mean CV results:

- **Linear Regression** — Mean R²: **0.6349**, Mean RMSE: **69727.01**
- **Random Forest Regressor** — Mean R²: **0.8190**, Mean RMSE: **49067.46**
- **Gradient Boosting Regressor** — Mean R²: **0.7762**, Mean RMSE: **54560.69**

## Feature Importance

The most important features were:

- **median_income**
- **latitude**
- **longitude**

Both Random Forest and Gradient Boosting showed that income and geographic location were the strongest predictors of house value.

## Visualization

The project also includes an **Actual vs Predicted** plot for model evaluation.

This helps visualize:
- how closely predictions match real values
- where the model makes larger errors
- the price cap effect visible in the dataset

## Conclusion

Random Forest Regressor achieved the best performance in both:
- train/test split evaluation
- cross-validation evaluation

Gradient Boosting Regressor performed better than Linear Regression, but worse than Random Forest.

The results show that tree-based models are much more effective than a simple linear model for this dataset.

## How to run

```bash
python california_housing_regression.py
