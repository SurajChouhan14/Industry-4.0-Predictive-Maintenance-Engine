"""
Remaining Useful Life (RUL) Telemetry Regression Engine.
Predicts the exact number of operational hours/cycles remaining before catastrophic failure
using Gradient Boosting Regressors on time-series telemetry sensor features.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class RemainingUsefulLifeEngine:
    """
    Trains and evaluates RUL regression models on industrial telemetry streams.
    """

    def __init__(self, n_estimators=150, max_depth=4, learning_rate=0.08):
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        )
        self.feature_cols = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits gradient boosting regressor on telemetry features.
        """
        self.feature_cols = list(X.columns)
        self.model.fit(X, y)
        return self

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Calculates Root Mean Squared Error (RMSE), MAE, and R² accuracy.
        """
        preds = self.model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        return {
            "rmse_cycles": rmse,
            "mae_cycles": mae,
            "r2_score": r2,
            "predictions": preds
        }

    def get_feature_importances(self) -> dict:
        """
        Extracts sensor feature importance rankings.
        """
        imps = self.model.feature_importances_
        return {col: float(imp) for col, imp in sorted(zip(self.feature_cols, imps), key=lambda x: -x[1])}
