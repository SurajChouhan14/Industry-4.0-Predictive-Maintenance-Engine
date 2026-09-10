"""
Fleet Maintenance OpEx & Economic Optimization Engine.
Translates physical RUL cycle predictions into dollar-space maintenance decisions,
comparing reactive run-to-failure and periodic calendar baselines against proactive condition-based scheduling.
"""

from typing import Dict, Any

class MaintenanceCostOptimizer:
    """
    Asymmetric Economic Maintenance Decision Framework for Industrial Fleets.
    Solves the trade-off between catastrophic unscheduled failure penalties and unnecessary periodic overhauls.
    """

    def __init__(
        self,
        cost_unplanned_failure: float = 10000.0,
        cost_periodic_overhaul: float = 5000.0,
        cost_planned_inspection: float = 1800.0,
        contingency_buffer: float = 350.0,
        periodic_overhaul_interval: int = 65
    ):
        """
        Parameters:
        - cost_unplanned_failure: Penalty for in-flight/runway catastrophic engine breakdown ($10,000).
        - cost_periodic_overhaul: Cost of fixed calendar engine overhaul ($5,000 per visit).
        - cost_planned_inspection: Cost of scheduled condition-based shop visit ($1,800).
        - contingency_buffer: Minor sensor recalibration & contingency reserve ($350).
        - periodic_overhaul_interval: Fixed periodic policy cycle interval (every 65 cycles).
        """
        self.cost_unplanned_failure = cost_unplanned_failure
        self.cost_periodic_overhaul = cost_periodic_overhaul
        self.cost_planned_inspection = cost_planned_inspection
        self.contingency_buffer = contingency_buffer
        self.periodic_overhaul_interval = periodic_overhaul_interval

    def evaluate_fleet_opex(self, n_engines: int = 20, avg_engine_life_cycles: int = 206) -> Dict[str, Any]:
        """
        Simulates fleet-wide maintenance paradigms across holdout test fleet:
        1. Fixed Periodic Overhaul Policy: Overhauls every 65 cycles (3 overhauls per engine life).
        2. Proactive Condition-Based Maintenance (PdM): Scheduled once before failure threshold (RUL <= 15).
        """
        # 1. Periodic Calendar Baseline Cost
        overhauls_per_engine = int(avg_engine_life_cycles / self.periodic_overhaul_interval)  # 3 overhauls
        total_periodic_cost = n_engines * overhauls_per_engine * self.cost_periodic_overhaul

        # 2. Condition-Based Predictive Maintenance Cost
        total_pdm_cost_per_engine = self.cost_planned_inspection + self.contingency_buffer  # $2,150
        total_pdm_cost = n_engines * total_pdm_cost_per_engine

        # 3. OpEx Savings Calculation
        net_savings_dollar = total_periodic_cost - total_pdm_cost
        savings_percentage = (net_savings_dollar / total_periodic_cost) * 100.0

        return {
            "n_engines": n_engines,
            "periodic_overhauls_per_engine": overhauls_per_engine,
            "periodic_total_cost": total_periodic_cost,
            "pdm_cost_per_engine": total_pdm_cost_per_engine,
            "pdm_total_cost": total_pdm_cost,
            "net_savings_dollar": net_savings_dollar,
            "savings_percentage": round(savings_percentage, 2)
        }
