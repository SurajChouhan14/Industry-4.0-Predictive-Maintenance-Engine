"""
Automated Test Suite: Industry 4.0 Predictive Maintenance Engine.
Verifies authentic AI4I dataset integrity, feature engineering, discriminatory metrics, and economic optimization.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from ai4i_data_loader import AI4IDataLoader
from failure_classifier import MachineFailureClassifier
from maintenance_cost_optimizer import MaintenanceCostOptimizer

class TestPredictiveMaintenanceEngine(unittest.TestCase):
    """Unit and integration tests for AI4I predictive maintenance engine."""

    @classmethod
    def setUpClass(cls):
        cls.loader = AI4IDataLoader()
        cls.raw_df = cls.loader.load_data()
        cls.processed_df = cls.loader.engineer_features()
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = cls.loader.get_train_test_split(
            test_size=0.20, random_state=42
        )

    def test_01_dataset_integrity(self):
        """Verifies exactly 10,000 records and 3.39% failure class rate."""
        self.assertEqual(len(self.raw_df), 10000, "Dataset must have exactly 10,000 records.")
        failure_count = self.raw_df['Machine failure'].sum()
        self.assertEqual(failure_count, 339, "Dataset must contain exactly 339 machine failures.")
        self.assertAlmostEqual(failure_count / 10000.0, 0.0339, places=4)

    def test_02_feature_engineering(self):
        """Verifies thermodynamic dissipation and mechanical power features are engineered correctly."""
        self.assertIn('thermal_dissipation_K', self.processed_df.columns)
        self.assertIn('power_dissipation_kW', self.processed_df.columns)
        self.assertIn('overstrain_torque_wear', self.processed_df.columns)
        
        # Verify non-null and physically valid values
        self.assertFalse(self.processed_df['thermal_dissipation_K'].isnull().any())
        self.assertTrue((self.processed_df['power_dissipation_kW'] > 0).all())

    def test_03_classifier_performance(self):
        """Verifies out-of-sample ROC-AUC >= 90% and PR-AUC >= 80% on imbalanced failure classes."""
        clf = MachineFailureClassifier(
            n_estimators=200, learning_rate=0.06, max_depth=4, random_state=42
        )
        clf.fit(self.X_train, self.y_train)
        metrics = clf.evaluate(self.X_test, self.y_test)
        
        self.assertGreaterEqual(metrics['roc_auc'], 0.90, "ROC-AUC must exceed 90.0%.")
        self.assertGreaterEqual(metrics['pr_auc'], 0.80, "PR-AUC (AUPRC) must exceed 80.0%.")

    def test_04_maintenance_cost_optimization(self):
        """Verifies predictive maintenance delivers significant OpEx savings over reactive baseline."""
        clf = MachineFailureClassifier(n_estimators=200, learning_rate=0.06, max_depth=4, random_state=42)
        clf.fit(self.X_train, self.y_train)
        y_probs = clf.predict_proba(self.X_test)
        
        optimizer = MaintenanceCostOptimizer(c_planned=500.0, c_unplanned=10000.0, c_inspection=100.0)
        results = optimizer.evaluate_threshold_economics(self.y_test.values, y_probs)
        
        self.assertGreaterEqual(results['cost_reduction_vs_reactive_pct'], 70.0, "Cost savings vs reactive must exceed 70%.")
        self.assertGreaterEqual(results['cost_reduction_vs_periodic_pct'], 70.0, "Cost savings vs periodic must exceed 70%.")

if __name__ == '__main__':
    unittest.main()
