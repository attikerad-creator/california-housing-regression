import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 1. Wczytanie danych 
df = pd.read_csv("https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv")
df = df.drop(columns=["ocean_proximity"])
df = df.dropna()

X = df.drop(columns=["median_house_value"])
y = df["median_house_value"]

print("X shape:", X.shape)
print("y shape:", y.shape)
print("\nPierwsze 5 wierszy X:")
print(X.head())

# 2. Podzial na train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

r2_lr = r2_score(y_test, y_pred_lr)
rmse_lr = mean_squared_error(y_test, y_pred_lr, squared=False)

print("\n=== Linear Regression ===")
print("R2:", round(r2_lr, 4))
print("RMSE:", round(rmse_lr, 4))

# 4. Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)

print("\n=== Random Forest Regressor ===")
print("R2:", round(r2_rf, 4))
print("RMSE:", round(rmse_rf, 4))

# 5. Gradient Boosting Regressor
gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(X_train, y_train)
y_pred_gbr = gbr.predict(X_test)

gbr_r2 = r2_score(y_test, y_pred_gbr)
gbr_rmse = mean_squared_error(y_test, y_pred_gbr, squared=False)

print("\n=== Gradient Boosting Regressor ===")
print(f"R2: {gbr_r2:.4f}")
print(f"RMSE: {gbr_rmse:.2f}")

print("\nWniosek:")
print("Random Forest osiągnął lepsze wyniki niż Linear Regression.")

# 6. Cross-validate
cv_scoring = {"r2": "r2", "rmse": "neg_root_mean_squared_error"}
models = {"LinearRegresion": lr, "RandomForestRegressor": rf, "GradientBoostingReressor": gbr}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n=== Cross-validate (5-fold) ===")
for name, model in models.items():
	cv_results = cross_validate(model, X, y, cv=kf, scoring=cv_scoring)
	mean_r2 = cv_results["test_r2"].mean()
	mean_rmse = -cv_results["test_rmse"].mean()
	print(f"\n{name}:")
	print(f"Mean R2: {mean_r2:.4f}")
	print(f"Mean RMSE: {mean_rmse:.2f}")

# 7. Feature importance dla Random Forest
importances = rf.feature_importances_

importance_df = pd.DataFrame({"feature": X.columns, "importance": importances})
importance_df = importance_df.sort_values(by="importance", ascending=False)

print("\n=== Feature Importance (Random Forest) ===")
print(importance_df)

# 8. Feature importance dla Gradient Boosting
gbr_importance = pd.DataFrame({"feature": X.columns, "importance": gbr.feature_importances_}).sort_values(by="importance", ascending=False)

print("\n=== Feature Importance (Gradient Boosting) ===")
print(gbr_importance)

# 9. Wykres feature importance
plt.figure(figsize=(10, 6))
plt.bar(importance_df["feature"], importance_df["importance"])
plt.xticks(rotation=45)
plt.title("feature Importance - Random Forest")
plt.xlabel("Feature")
plt.ylabel("importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# 10. Wykres actual vs predicted dla Random Forest
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred_rf, alpha=0.3)

plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Actual vs Predicted - Random Forest")

min_val = min(y_test.min(), y_pred_rf.min())
max_val = max(y_test.max(), y_pred_rf.max())
plt.plot([min_val, max_val], [min_val, max_val])

plt.tight_layout()
plt.savefig("actual_vs_predicted_rf.png")
plt.show()

# 11. Wykres actual vs predicted dla Gradient Boosting
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_gbr, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Gradient Boosting: Actual vs Predicted")
plt.show()

