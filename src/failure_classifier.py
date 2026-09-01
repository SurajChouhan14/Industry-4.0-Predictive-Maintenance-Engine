"""
Cost-Sensitive Failure Classifier & Multi-Modal Machine Breakdown Prediction.
Evaluates ROC-AUC, PR-AUC (AUPRC), and multi-class failure modes on AI4I 2020 dataset.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    classification_report, confusion_matrix
)

class MachineFailureClassifier:
    """
    Gradient Boosted Machine Failure Classifier optimized for imbalanced sensor telemetry (3.39% failure rate).
    """

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.06,
        max_depth: int = 4,
        min_samples_split: int = 6,
        min_samples_leaf: int = 4,
        random_state: int = 42
    ):
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=0.85,
            random_state=random_state
        )
        self.is_fitted = False
        self.feature_names = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'MachineFailureClassifier':
        """Fits the gradient boosted classifier on telemetry training features."""
        self.feature_names = list(X_train.columns)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns failure probability estimates [P(Normal), P(Failure)]."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba.")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.50) -> np.ndarray:
        """Returns binary predictions based on decision threshold."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Calculates comprehensive out-of-sample discriminatory metrics."""
        probs = self.predict_proba(X_test)
        
        roc_auc = float(roc_auc_score(y_test, probs))
        pr_auc = float(average_precision_score(y_test, probs))
        
        # Binary metrics at standard threshold (0.50)
        preds_50 = (probs >= 0.50).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds_50).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Feature importances
        importances = dict(zip(self.feature_names, [float(x) for x in self.model.feature_importances_]))
        
        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': {'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)},
            'feature_importances': importances
        }
