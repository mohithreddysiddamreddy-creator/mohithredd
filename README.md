# House Price Prediction

A small machine-learning example that predicts house prices using linear
regression.

## Dataset

The sample dataset contains:

- `area`: house area in square feet
- `bedrooms`: number of bedrooms
- `age`: house age in years
- `price`: target price in dataset units

The data is defined in [`src/housing_data.py`](src/housing_data.py).

## Run locally

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the training program:

```bash
python src/train.py
```

The program prints a sample prediction and evaluates the model using RMSE and
MAE:

```text
Estimated price: 40.5 (dataset price units)
RMSE: 1.00
MAE: 0.77
```

## GitHub Actions

The workflow in [`.github/workflows/train.yml`](.github/workflows/train.yml)
runs the training program on pushes and pull requests that change files under
`src/**`.
