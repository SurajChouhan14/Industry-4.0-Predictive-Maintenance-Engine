"""
Automated Unit and Integration Test Suite: Turbofan Predictive Maintenance Engine.
Verifies NASA C-MAPSS dataset integrity, 21 sensor channels, RUL clipping, MAE convergence, and OpEx savings.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from cmapss_data_loader import CMAPSSDataLoader
from rul_predictor import TurbofanRULPredictor
from maintenance_cost_optimizer import MaintenanceCostOptimizer

class TestTurbofanPredictiveMaintenanceEngine(unittest.TestCase):
    """Unit and integration tests for NASA C-MAPSS prognostic engine."""

    @classmethod
    def setUpClass(cls):
        cls.loader = CMAPSSDataLoader(max_rul_clip=125, rolling_window=5)
        cls.raw_df = cls.loader.load_raw_data()
        cls.processed_df = cls.loader.engineer_features()
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = cls.loader.get_train_test_split(
            test_size=0.20, random_state=51
        )
        cls.predictor = TurbofanRULPredictor(n_estimators=100, learning_rate=0.07, max_depth=3, random_state=51)
        cls.predictor.fit(cls.X_train, cls.y_train)
        cls.metrics = cls.predictor.evaluate(cls.X_test, cls.y_test)

    def test_01_dataset_integrity(self):
        """Verifies authentic NASA C-MAPSS train_FD001 records (100 engines, 20,631 cycles)."""
        self.assertEqual(len(self.raw_df), 20631, "FD001 dataset must contain exactly 20,631 cycle records.")
        self.assertEqual(self.raw_df["engine_id"].nunique(), 100, "Dataset must contain exactly 100 jet engines.")

    def test_02_sensor_channels(self):
        """Verifies exactly 21 telemetry channels ingested and dead sensors properly identified."""
        self.assertEqual(len(self.loader.SENSOR_METADATA), 21, "Must define 21 sensor channels.")
        self.assertEqual(len(self.loader.active_sensors), 14, "Must identify exactly 14 active degradation sensors.")

    def test_03_rul_piecewise_clipping(self):
        """Verifies piece-wise linear clipping bounds RUL target at 125 cycles."""
        self.assertLessEqual(self.processed_df["rul_clipped"].max(), 125.0)
        self.assertGreaterEqual(self.processed_df["rul_clipped"].min(), 0.0)

    def test_04_gbrt_prognostic_accuracy(self):
        """Verifies out-of-sample MAE satisfies resume target threshold (<= 15.5 cycles)."""
        self.assertLessEqual(self.metrics["mae"], 15.50, f"MAE {self.metrics['mae']} must be <= 15.5 cycles.")
        self.assertGreaterEqual(self.metrics["r2"], 0.70, "R2 score must exceed 0.70.")

    def test_05_fleet_opex_savings(self):
        """Verifies condition-based maintenance delivers >= 85.0% OpEx savings over periodic overhauls."""
        optimizer = MaintenanceCostOptimizer()
        cost_metrics = optimizer.evaluate_fleet_opex(n_engines=20, avg_engine_life_cycles=206)
        self.assertGreaterEqual(cost_metrics["savings_percentage"], 85.0, "OpEx reduction must exceed 85.0%.")
        self.assertAlmostEqual(cost_metrics["savings_percentage"], 85.67, delta=0.5)

if __name__ == "__main__":
    unittest.main()
