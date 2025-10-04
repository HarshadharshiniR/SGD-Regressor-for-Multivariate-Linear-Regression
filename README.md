# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset and split it into independent variables (features) and dependent variables (Price, Occupants).
2.Preprocess the data by standardizing the features for better SGD convergence.
3.Initialize the SGDRegressor and train it on the scaled training data.
4.Use the trained model to predict house price and occupants, then evaluate using error metrics.

## Program:
```
# Ex 04 - Multivariate Linear Regression Model Using SGD
# Aim: To predict the house price and number of occupants using SGD Regressor.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
data = {
    "Area": [1200, 1500, 1800, 2000, 2200, 2500],
    "Bedrooms": [2, 3, 3, 4, 4, 5],
    "Age": [5, 10, 8, 12, 6, 15],
    "Price": [200000, 250000, 270000, 300000, 320000, 400000],
    "Occupants": [3, 4, 4, 5, 6, 7]
}
df = pd.DataFrame(data)

# Independent variables (features)
X = df[["Area", "Bedrooms", "Age"]]

# Dependent variables (targets: Price, Occupants)
y_price = df["Price"]
y_occ = df["Occupants"]

# Split into training and test sets
X_train, X_test, y_price_train, y_price_test, y_occ_train, y_occ_test = train_test_split(
    X, y_price, y_occ, test_size=0.3, random_state=42
)

# Feature scaling (important for SGD)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------
# Model for predicting Price
sgd_price = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.01, learning_rate='constant')
sgd_price.fit(X_train_scaled, y_price_train)

# Model for predicting Occupants
sgd_occ = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.01, learning_rate='constant')
sgd_occ.fit(X_train_scaled, y_occ_train)

# -------------------
# Predictions
y_price_pred = sgd_price.predict(X_test_scaled)
y_occ_pred = sgd_occ.predict(X_test_scaled)

# -------------------
# Evaluation
print("House Price Prediction:")
print("MSE:", mean_squared_error(y_price_test, y_price_pred))
print("R² Score:", r2_score(y_price_test, y_price_pred))

print("\nOccupants Prediction:")
print("MSE:", mean_squared_error(y_occ_test, y_occ_pred))
print("R² Score:", r2_score(y_occ_test, y_occ_pred))

# -------------------
# Example new prediction
new_house = np.array([[2100, 4, 9]])  # [Area, Bedrooms, Age]
new_house_scaled = scaler.transform(new_house)

predicted_price = sgd_price.predict(new_house_scaled)
predicted_occ = sgd_occ.predict(new_house_scaled)

print("\nPredicted House Price:", round(predicted_price[0], 2))
print("Predicted Number of Occupants:", round(predicted_occ[0], 0))


```

## Output:


<img width="470" height="224" alt="485015631-2732b27b-1340-41f4-9a46-f282c3cfce39" src="https://github.com/user-attachments/assets/62fc726b-b860-473a-909d-b888398ef0d1" />

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
