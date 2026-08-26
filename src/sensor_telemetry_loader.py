"""
Industrial Sensor Telemetry Data Loader.
Primary  : NASA CMAPSS Turbofan Engine Degradation Dataset - FD001 subset.
           100 training engines, 21 sensor channels, 3 operating condition channels.
           Single fault mode (HPC degradation), single operating condition.
Fallback : Weibull multi-sensor simulation if data absent or download fails.

Data source : ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
Reference   : Saxena A. et al. (2008) "Damage Propagation Modeling for Aircraft Engine
              Run-to-Failure Simulation". 2008 IEEE PHM Conference.

RUL computation:
  Piece-wise linear: RUL = max_cycle_per_engine - current_cycle (clipped at 125 cycles per PHM benchmark standards)
  failure_imminent = 1 if RUL <= 30 cycles (standard threshold in PHM literature)

Sensors used downstream:
  vibration_rms (s2), temperature_c (s3), hydraulic_pressure_bar (s4), acoustic_emission_db (s7),
  along with rolling statistical features (rolling mean, rolling std over 5 cycles).
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


class IndustrialTelemetryLoader:
    """
    Loads NASA CMAPSS FD001 run-to-failure turbofan engine degradation data.
    Engineers RUL, rolling degradation features, and failure_imminent labels.
    """

    LOCAL_FILE = "train_FD001.txt"

    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def generate_telemetry_dataset(self, n_machines=250, max_cycles=350, seed=42):
        """
        Entry point called by run_pipeline.py.
        Routes to NASA CMAPSS FD001; auto-downloads if absent; Weibull fallback if offline.
        """
        local_path = os.path.join(self.data_dir, self.LOCAL_FILE)

        if os.path.exists(local_path):
            return self._load_and_engineer(local_path)

        if _URLLIB_OK:
            try:
                print("      Downloading NASA CMAPSS FD001 from GitHub mirror...")
                _urllib.urlretrieve(_CMAPSS_URL, local_path)
                return self._load_and_engineer(local_path)
            except Exception as e:
                print(f"      WARNING: Download failed ({e}). Using Weibull simulation fallback.")

        return self._generate_weibull_simulation(n_machines, max_cycles, seed)

    def _load_and_engineer(self, path):
        """
        Loads raw CMAPSS space-separated file and engineers:
          - RUL (piece-wise linear): max_cycle_per_engine - current_cycle
          - RUL Clipped at 125 cycles (standard in Prognostics & Health Management literature)
          - Rolling mean and rolling standard deviation over 5-cycle windows
        """
        df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
        df = df.dropna(axis=1, how="all")
        n_cols = df.shape[1]
        df.columns = _ALL_COLS[:n_cols]

        # RUL engineering
        max_cyc = df.groupby("engine_id")["cycle"].max().rename("max_cycle")
        df = df.merge(max_cyc, on="engine_id")
        df["rul_cycles"] = df["max_cycle"] - df["cycle"]
        df["rul_clipped"] = df["rul_cycles"].clip(upper=125)
        df["failure_imminent"] = (df["rul_cycles"] <= 30).astype(int)
        df = df.drop(columns=["max_cycle"])

        # Descriptive column names for 4 key sensors
        rename_map = {
            "s2": "vibration_rms",
            "s3": "temperature_c",
            "s4": "hydraulic_pressure_bar",
            "s7": "acoustic_emission_db",
        }
        df = df.rename(columns=rename_map)
        df["machine_id"] = df["engine_id"].apply(lambda x: f"ENG_{int(x):04d}")

        # Compute rolling degradation features per engine
        base_features = ["vibration_rms", "temperature_c", "hydraulic_pressure_bar", "acoustic_emission_db"]
        for feat in base_features:
            df[f"{feat}_roll_mean"] = df.groupby("engine_id")[feat].transform(lambda x: x.rolling(5, min_periods=1).mean())
            df[f"{feat}_roll_std"] = df.groupby("engine_id")[feat].transform(lambda x: x.rolling(5, min_periods=1).std()).fillna(0.0)

        n_eng = df["machine_id"].nunique()
        print(f"      Real NASA CMAPSS FD001: {len(df):,} rows | "
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
                records.append({
                    "machine_id": f"MACH_{m_id:04d}", "cycle": cycle,
                    "vibration_rms": round(bv + 9.5*d + np.random.normal(0, 0.3), 3),
                    "temperature_c": round(bt + 42.0*d + np.random.normal(0, 1.5), 2),
                    "hydraulic_pressure_bar": round(bp - 2.8*d + np.random.normal(0, 0.1), 2),
                    "acoustic_emission_db": round(45.0 + 50.0*d + np.random.normal(0, 2.0), 1),
                    "rul_cycles": fail_cycle - cycle,
                    "rul_clipped": min(125, fail_cycle - cycle),
                    "failure_imminent": 1 if fail_cycle - cycle <= 30 else 0,
                })
        df_sim = pd.DataFrame(records)
        base_features = ["vibration_rms", "temperature_c", "hydraulic_pressure_bar", "acoustic_emission_db"]
        for feat in base_features:
            df_sim[f"{feat}_roll_mean"] = df_sim.groupby("machine_id")[feat].transform(lambda x: x.rolling(5, min_periods=1).mean())
            df_sim[f"{feat}_roll_std"] = df_sim.groupby("machine_id")[feat].transform(lambda x: x.rolling(5, min_periods=1).std()).fillna(0.0)

        print(f"      FALLBACK: Weibull simulation ({n_machines} machines). Network unavailable.")
        return df_sim
