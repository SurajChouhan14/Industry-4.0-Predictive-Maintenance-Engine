"""
Fleet Maintenance Cost Optimizer & Economic Loss Minimization.
Calculates OpEx savings of Condition-Based Maintenance (PdM) vs Reactive Failure & Periodic Overhaul.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

class MaintenanceCostOptimizer:
    """
    Economic loss matrix evaluator for predictive maintenance:
    - Planned Preventative Maintenance Cost (C_planned): $500 per machine
    - Unplanned Catastrophic Breakdown Cost (C_unplanned): $10,000 (downtime + emergency repairs)
    - False Alarm Inspection Cost (C_inspection): $100
    """

    def __init__(
        self,
        c_planned: float = 500.0,
        c_unplanned: float = 10000.0,
        c_inspection: float = 100.0
    ):
        self.c_planned = c_planned
        self.c_unplanned = c_unplanned
        self.c_inspection = c_inspection

    def evaluate_threshold_economics(
        self, y_true: np.ndarray, y_probs: np.ndarray, thresholds: np.ndarray = None
    ) -> Dict[str, Any]:
        """Finds the optimal decision threshold T* minimizing total fleet operational losses."""
        if thresholds is None:
            thresholds = np.linspace(0.01, 0.99, 100)

        best_threshold = 0.50
        min_cost = float('inf')
        total_machines = len(y_true)
        actual_failures = int(np.sum(y_true))

        # Baseline 1: Pure Reactive Run-to-Failure (No prediction, all failures incur C_unplanned)
        reactive_baseline_cost = actual_failures * self.c_unplanned

        # Baseline 2: Fixed Periodic Overhaul (Overhaul every machine periodically -> 100% * C_planned)
        periodic_baseline_cost = total_machines * self.c_planned

        for t in thresholds:
            y_pred = (y_probs >= t).astype(int)
            
            # Confusion matrix elements
            tp = np.sum((y_true == 1) & (y_pred == 1))  # Prevented failures (Planned service)
            fn = np.sum((y_true == 1) & (y_pred == 0))  # Missed failures (Catastrophic breakdown)
            fp = np.sum((y_true == 0) & (y_pred == 1))  # False alarms (Unnecessary inspection)
            tn = np.sum((y_true == 0) & (y_pred == 0))  # True normal (Zero cost)

            total_cost = (tp * self.c_planned) + (fn * self.c_unplanned) + (fp * self.c_inspection)

            if total_cost < min_cost:
                min_cost = total_cost
                best_threshold = float(t)
                best_breakdown = {
                    'tp_prevented': int(tp),
                    'fn_catastrophic': int(fn),
                    'fp_inspections': int(fp),
                    'total_cost': float(total_cost)
                }

        cost_reduction_vs_reactive = (
            (reactive_baseline_cost - min_cost) / reactive_baseline_cost
        ) * 100.0 if reactive_baseline_cost > 0 else 0.0

        cost_reduction_vs_periodic = (
            (periodic_baseline_cost - min_cost) / periodic_baseline_cost
        ) * 100.0 if periodic_baseline_cost > 0 else 0.0

        return {
            'optimal_threshold': best_threshold,
            'optimal_cost': min_cost,
            'reactive_baseline_cost': float(reactive_baseline_cost),
            'periodic_baseline_cost': float(periodic_baseline_cost),
            'cost_reduction_vs_reactive_pct': cost_reduction_vs_reactive,
            'cost_reduction_vs_periodic_pct': cost_reduction_vs_periodic,
            'optimal_breakdown': best_breakdown
        }
