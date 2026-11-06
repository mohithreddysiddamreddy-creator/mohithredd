"""Train a linear-regression model with the sample housing dataset."""

import pandas as pd
from housing_data import load_data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train_model() -> LinearRegression:
    """Fit a model using area, bedrooms, and age as features."""
    data = load_data()
    features = data[["area", "bedrooms", "age"]]
    prices = data["price"]

    model = LinearRegression()
    model.fit(features, prices)
    return model


if __name__ == "__main__":
    data = load_data()
    model = train_model()
    features = data[["area", "bedrooms", "age"]]
    actual_prices = data["price"]
    predicted_prices = model.predict(features)
    rmse = mean_squared_error(actual_prices, predicted_prices) ** 0.5
    mae = mean_absolute_error(actual_prices, predicted_prices)

    new_house = pd.DataFrame([[1300, 3, 5]], columns=["area", "bedrooms", "age"])
    prediction = model.predict(new_house)[0]
    print(f"Estimated price: {prediction:.1f} (dataset price units)")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
