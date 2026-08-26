"""
Automated Unit Test Suite for Industry 4.0 Predictive Maintenance Engine.
Verifies Sensor Telemetry Ingestion, RUL Regression Training, Feature Importance, and Maintenance OpEx Optimization.
"""

import unittest
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.sensor_telemetry_loader import IndustrialTelemetryLoader
from src.rul_regressor import RemainingUsefulLifeEngine
from src.maintenance_cost_optimizer import MaintenanceOpExOptimizer


class TestPredictiveMaintenanceEngine(unittest.TestCase):
    """
    Unit test cases for industrial predictive maintenance and RUL regression engine.
    """

    def setUp(self):
        self.loader = IndustrialTelemetryLoader(data_dir="data")
        self.rul_engine = RemainingUsefulLifeEngine(n_estimators=50, max_depth=3)
        self.cost_optimizer = MaintenanceOpExOptimizer(reactive_cost=12000.0, periodic_cost=2500.0, pdm_cost=1200.0)
        self.df = self.loader.generate_telemetry_dataset()

    def test_telemetry_data_loading(self):
        """Verify telemetry stream ingestion, engine count, and rolling features."""
        self.assertGreaterEqual(len(self.df), 1000)
        self.assertIn("machine_id", self.df.columns)
        self.assertIn("cycle", self.df.columns)
        self.assertIn("vibration_rms", self.df.columns)
        self.assertIn("vibration_rms_roll_mean", self.df.columns)
        self.assertIn("rul_cycles", self.df.columns)

    def test_rul_regressor_training_and_evaluation(self):
        """Verify RUL regression model fits and evaluates bounded positive errors."""
        feature_cols = ["vibration_rms", "temperature_c", "hydraulic_pressure_bar", "acoustic_emission_db"]
        sample_df = self.df.sample(500, random_state=42)
        X = sample_df[feature_cols]
        y = sample_df["rul_clipped"]

        self.rul_engine.fit(X, y)
        metrics = self.rul_engine.evaluate(X, y)

        self.assertIn("r2_score", metrics)
        self.assertIn("mae_cycles", metrics)
        self.assertIn("rmse_cycles", metrics)
        self.assertGreater(metrics["mae_cycles"], 0.0)
        self.assertLess(metrics["mae_cycles"], 50.0)

    def test_feature_importance_extraction(self):
        """Verify sensor feature importance attribution sums to 1.0."""
        feature_cols = ["vibration_rms", "temperature_c", "hydraulic_pressure_bar", "acoustic_emission_db"]
        sample_df = self.df.sample(500, random_state=42)
        self.rul_engine.fit(sample_df[feature_cols], sample_df["rul_clipped"])
        imps = self.rul_engine.get_feature_importances()

        self.assertEqual(len(imps), 4)
        np.testing.assert_allclose(sum(imps.values()), 1.0, rtol=1e-4)

    def test_maintenance_cost_optimization(self):
        """Verify OpEx simulation produces cost reduction vs reactive baseline."""
        test_df = self.df[self.df["machine_id"].isin(self.df["machine_id"].unique()[:10])].copy()
        test_df["rul_pred"] = test_df["rul_clipped"]

        res = self.cost_optimizer.simulate_fleet_costs(test_df, rul_pred_col="rul_pred")
        self.assertIn("total_pdm_cost_usd", res)
        self.assertIn("savings_vs_reactive_pct", res)
        self.assertGreater(res["savings_vs_reactive_pct"], 50.0)


if __name__ == '__main__':
    unittest.main()
