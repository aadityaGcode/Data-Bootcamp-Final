import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_linear_regression(X, y):
    """Train and return a LinearRegression model."""
    model = LinearRegression()
    model.fit(X, y)
    return model


def train_neural_network(X, y, hidden_layer_sizes=(10,), max_iter=500, random_state=0):
    """
    Train and return an MLPRegressor and its scaler.
    """
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    nn = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                      activation='relu', max_iter=max_iter,
                      random_state=random_state)
    nn.fit(X_scaled, y)
    return nn, scaler


def evaluate_model(model, X, y_true, scaler=None):
    """Return MAE, RMSE, and predictions for given model."""
    if scaler:
        X_input = scaler.transform(X)
    else:
        X_input = X
    y_pred = model.predict(X_input)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse, y_pred


def plot_predictions(years, actual, pred_lin, pred_nn=None):
    """Plot actual vs. predicted GDP per capita."""
    plt.figure(figsize=(10,6))
    plt.plot(years, actual, label='Actual', marker='o')
    plt.plot(years, pred_lin, label='Linear Model', marker='x')
    if pred_nn is not None:
        plt.plot(years, pred_nn, label='Neural Net', marker='s')
    plt.axvline(years[0], color='gray', linestyle='--', label='Test Start')
    plt.xlabel('Year')
    plt.ylabel('Real GDP per Capita (2017 USD)')
    plt.legend()
    plt.tight_layout()
    plt.show()
