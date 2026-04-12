import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
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

print("\nWniosek:")
print("Random Forest osiągnął lepsze wyniki niż Linear Regression.")
