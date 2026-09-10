"""
Gradient Boosted Remaining Useful Life (RUL) Regressor for Turbofan Prognostics.
Fits GBRT with piece-wise linear target formulation, evaluating out-of-sample MAE, RMSE, and R2.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any

class TurbofanRULPredictor:
    """
    Gradient Boosted Regression Tree (GBRT) prognostic estimator for turbofan RUL.
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.07, max_depth: int = 3, random_state: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.feature_importances_ = None
        self.is_fitted = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fits the GBRT regressor on sensor degradation features."""
        self.model.fit(X_train, y_train)
        self.feature_importances_ = dict(zip(X_train.columns, self.model.feature_importances_))
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generates RUL cycle predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before generating predictions.")
        return np.clip(self.model.predict(X), 0, 125)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Computes out-of-sample MAE, RMSE, R2, and top feature drivers."""
        preds = self.predict(X_test)
        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))

        sorted_imp = sorted(self.feature_importances_.items(), key=lambda x: x[1], reverse=True)

        return {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2": round(r2, 4),
            "predictions": preds,
            "top_features": sorted_imp[:6]
        }
