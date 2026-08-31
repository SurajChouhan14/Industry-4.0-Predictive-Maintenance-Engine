"""
NASA C-MAPSS Turbofan Engine Telemetry & Degradation Feature Loader.
Dataset  : NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) - FD001 subset.
           100 run-to-failure turbofan engines, 21 sensor telemetry channels, 3 operating setting channels.
           Single fault mode (High-Pressure Compressor / HPC degradation) under sea-level operating conditions.

Data Source: NASA Prognostics Center of Excellence (PCoE) Prognostic Data Repository
Reference  : Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008).
             "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation".
             IEEE International Conference on Prognostics and Health Management.

NASA C-MAPSS FD001 Gas-Path Sensor Nomenclature:
- Active / Informative Channels (14 sensors with degradation dynamics):
  • s2  : T24 - Total temperature at LPC outlet (°R)
  • s3  : T30 - Total temperature at HPC outlet (°R)
  • s4  : T50 - Total temperature at LPT outlet (°R)
  • s7  : P30 - Total pressure at HPC outlet (psia)
  • s8  : Nf  - Physical fan speed (rpm)
  • s9  : Nc  - Physical core speed (rpm)
  • s11 : Ps30 - Static pressure at HPC outlet (psia)
  • s12 : Phi - Ratio of fuel flow to Ps30 (pps/psi)
  • s13 : NRf - Corrected fan speed (rpm)
  • s14 : NRc - Corrected core speed (rpm)
  • s15 : BPR - Bypass Ratio
  • s17 : htBleed - Bleed Enthalpy
  • s20 : W31 - HPT coolant bleed (lbm/s)
  • s21 : W32 - LPT coolant bleed (lbm/s)
- Constant / Dead Channels in FD001 (7 sensors with zero variance):
  • s1 (T2), s5 (P2), s6 (P15), s10 (epr), s16 (farB), s18 (Nf_dmd), s19 (PCNfR_dmd)

RUL Formulation:
- Piece-wise linear: Target RUL = min(125, max_cycle_per_engine - current_cycle) per PHM benchmark standards.
- failure_imminent: Binary label (1 if RUL <= 30 cycles).
"""

import os
import numpy as np
import pandas as pd

try:
    import urllib.request as _urllib
    _URLLIB_OK = True
except ImportError:
    _URLLIB_OK = False

_CMAPSS_URL = ("https://raw.githubusercontent.com/hankroark/"
               "Turbofan-Engine-Degradation/master/CMAPSSData/train_FD001.txt")

_ALL_COLS = (["engine_id", "cycle", "op1", "op2", "op3"]
             + [f"s{i}" for i in range(1, 22)])

# 14 Active Informative Sensor Channels (Saxena et al. 2008)
ACTIVE_SENSOR_MAP = {
    "s2": "s2_T24_LPC_temp",
    "s3": "s3_T30_HPC_temp",
    "s4": "s4_T50_LPT_temp",
    "s7": "s7_P30_HPC_press",
    "s8": "s8_Nf_fan_speed",
    "s9": "s9_Nc_core_speed",
    "s11": "s11_Ps30_static_press",
    "s12": "s12_Phi_fuel_ratio",
    "s13": "s13_NRf_corr_fan_speed",
    "s14": "s14_NRc_corr_core_speed",
    "s15": "s15_BPR_bypass_ratio",
    "s17": "s17_htBleed_enthalpy",
    "s20": "s20_W31_HPT_bleed",
    "s21": "s21_W32_LPT_bleed"
}

DEAD_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]


class IndustrialTelemetryLoader:
    """
    Loads NASA C-MAPSS FD001 run-to-failure turbofan degradation data and engineers sensor temporal dynamics.
    """

    LOCAL_FILE = "train_FD001.txt"

    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def generate_telemetry_dataset(self, n_machines=250, max_cycles=350, seed=42):
        """
        Ingests real NASA C-MAPSS FD001 dataset; auto-downloads if absent; Weibull simulation fallback if offline.
        """
        local_path = os.path.join(self.data_dir, self.LOCAL_FILE)

        if os.path.exists(local_path):
            return self._load_and_engineer(local_path)

        if _URLLIB_OK:
            try:
                print("      Downloading NASA C-MAPSS FD001 from repository mirror...")
                _urllib.urlretrieve(_CMAPSS_URL, local_path)
                return self._load_and_engineer(local_path)
            except Exception as e:
                print(f"      WARNING: Download failed ({e}). Using Weibull simulation fallback.")

        return self._generate_weibull_simulation(n_machines, max_cycles, seed)

    def _load_and_engineer(self, path):
        """
        Parses raw NASA C-MAPSS FD001 space-separated stream and engineers:
        - Piece-wise linear RUL target clipped at 125 cycles
        - 5-cycle rolling temporal moments (rolling mean, rolling standard deviation) across 14 active sensors
        - failure_imminent indicator (RUL <= 30 cycles)
        """
        df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
        df = df.dropna(axis=1, how="all")
        n_cols = df.shape[1]
        df.columns = _ALL_COLS[:n_cols]

        # RUL target engineering
        max_cyc = df.groupby("engine_id")["cycle"].max().rename("max_cycle")
        df = df.merge(max_cyc, on="engine_id")
        df["rul_cycles"] = df["max_cycle"] - df["cycle"]
        df["rul_clipped"] = df["rul_cycles"].clip(upper=125)
        df["failure_imminent"] = (df["rul_cycles"] <= 30).astype(int)
        df = df.drop(columns=["max_cycle"])

        # Rename active informative sensors with scientific gas-path identities
        df = df.rename(columns=ACTIVE_SENSOR_MAP)
        df["machine_id"] = df["engine_id"].apply(lambda x: f"ENG_{int(x):04d}")

        # Compute 5-cycle rolling moments per turbofan engine
        renamed_active_cols = list(ACTIVE_SENSOR_MAP.values())
        for feat in renamed_active_cols:
            df[f"{feat}_roll_mean"] = df.groupby("engine_id")[feat].transform(lambda x: x.rolling(5, min_periods=1).mean())
            df[f"{feat}_roll_std"] = df.groupby("engine_id")[feat].transform(lambda x: x.rolling(5, min_periods=1).std()).fillna(0.0)

        n_eng = df["machine_id"].nunique()
        print(f"      Real NASA C-MAPSS FD001: {len(df):,} rows | "
              f"{n_eng} engines | Max RUL: {df['rul_cycles'].max()} cycles | "
              f"Failure-imminent rows: {df['failure_imminent'].sum():,}")
        return df

    def _generate_weibull_simulation(self, n_machines, max_cycles, seed):
        """Weibull multi-sensor degradation simulation fallback."""
        np.random.seed(seed)
        records = []
        for m_id in range(1, n_machines + 1):
            fail_cycle = int(np.clip(np.random.weibull(5.0) * 180 + 100, 120, max_cycles))
            bv = np.random.normal(2.5, 0.2)
            bt = np.random.normal(65.0, 3.0)
            bp = np.random.normal(5.0, 0.2)
            for cycle in range(1, fail_cycle + 1):
                d = (cycle / float(fail_cycle)) ** 3.0
                rec = {
                    "machine_id": f"ENG_{m_id:04d}", "cycle": cycle,
                    "rul_cycles": fail_cycle - cycle,
                    "rul_clipped": min(125, fail_cycle - cycle),
                    "failure_imminent": 1 if fail_cycle - cycle <= 30 else 0,
                }
                for s_orig, s_name in ACTIVE_SENSOR_MAP.items():
                    rec[s_name] = round(100.0 + 20.0 * d + np.random.normal(0, 1.0), 3)
                records.append(rec)

        df_sim = pd.DataFrame(records)
        for feat in list(ACTIVE_SENSOR_MAP.values()):
            df_sim[f"{feat}_roll_mean"] = df_sim.groupby("machine_id")[feat].transform(lambda x: x.rolling(5, min_periods=1).mean())
            df_sim[f"{feat}_roll_std"] = df_sim.groupby("machine_id")[feat].transform(lambda x: x.rolling(5, min_periods=1).std()).fillna(0.0)

        print(f"      FALLBACK: Synthetic Weibull simulation ({n_machines} engines).")
        return df_sim
