"""
Industrial Maintenance OpEx & Downtime Loss Optimization Engine.
Compares 3 maintenance strategies:
1. Run-to-Failure (Reactive): Catastrophic downtime cost ($12,000 / incident + line shutdown)
2. Fixed-Schedule (Periodic): High preventative cost ($2,500 / overhaul every 50 cycles)
3. Predictive Maintenance (PdM): Prescriptive repair when RUL <= 25 cycles ($1,200 / scheduled repair)
"""

import numpy as np
import pandas as pd


class MaintenanceOpExOptimizer:
    """
    Quantifies capital expenditure and operational downtime savings across maintenance paradigms.
    """

    def __init__(self, reactive_cost=12000.0, periodic_cost=2500.0, pdm_cost=1200.0):
        self.reactive_cost = reactive_cost # Catastrophic replacement + plant stoppage
        self.periodic_cost = periodic_cost # Routine scheduled replacement (often premature)
        self.pdm_cost = pdm_cost           # Planned JIT component swap

    def simulate_fleet_costs(self, df: pd.DataFrame, rul_pred_col: str) -> dict:
        """
        Simulates total annual fleet maintenance expenditure across 250 industrial assets.
        """
        machines = df["machine_id"].unique()
        n_fleet = len(machines)

        # 1. Reactive Cost (Every machine runs until failure)
        total_reactive_cost = n_fleet * self.reactive_cost

        # 2. Periodic Cost (Overhaul every 60 cycles regardless of actual condition)
        total_cycles = len(df)
        periodic_interventions = total_cycles // 60
        total_periodic_cost = periodic_interventions * self.periodic_cost

        # 3. Predictive Maintenance Cost (Intervene once when predicted RUL <= 25)
        # Intercept before failure (planned maintenance)
        pdm_interventions = 0
        missed_failures = 0

        for m in machines:
            m_df = df[df["machine_id"] == m]
            # Check if alert was triggered before actual failure
            triggered = m_df[m_df[rul_pred_col] <= 25]
            if len(triggered) > 0:
                first_trigger_cycle = triggered.iloc[0]["cycle"]
                actual_fail_cycle = m_df["cycle"].max()
                if first_trigger_cycle <= actual_fail_cycle:
                    pdm_interventions += 1
                else:
                    missed_failures += 1
            else:
                missed_failures += 1

        total_pdm_cost = (pdm_interventions * self.pdm_cost) + (missed_failures * self.reactive_cost)
        savings_vs_reactive = float(((total_reactive_cost - total_pdm_cost) / total_reactive_cost) * 100.0)
        savings_vs_periodic = float(((total_periodic_cost - total_pdm_cost) / total_periodic_cost) * 100.0)

        return {
            "fleet_size": n_fleet,
            "total_reactive_cost_usd": float(total_reactive_cost),
            "total_periodic_cost_usd": float(total_periodic_cost),
            "total_pdm_cost_usd": float(total_pdm_cost),
            "pdm_interventions": pdm_interventions,
            "savings_vs_reactive_pct": savings_vs_reactive,
            "savings_vs_periodic_pct": savings_vs_periodic
        }
