"""
NASA C-MAPSS Turbofan Jet Engine Degradation Data Loader & Feature Engineering Engine.
Ingests authentic FD001 telemetry (100 engines, 20,631 cycles), filters dead sensor channels,
computes piece-wise linear clipped Remaining Useful Life (RUL), and builds rolling degradation features.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict

class CMAPSSDataLoader:
    """
    Ingests and transforms authentic NASA C-MAPSS turbofan engine sensor telemetry.
    Nomenclature follows Saxena et al. (PHM 2008) commercial modular aero-propulsion simulation.
    """

    SENSOR_METADATA: Dict[str, str] = {
        "s1": "T2 - Total temperature at fan inlet [deg R]",
        "s2": "T24 - Total temperature at LPC outlet [deg R]",
        "s3": "T30 - Total temperature at HPC outlet [deg R]",
        "s4": "T50 - Total temperature at LPT outlet [deg R]",
        "s5": "P2 - Pressure at fan inlet [psia]",
        "s6": "P15 - Total pressure in bypass-duct [psia]",
        "s7": "P30 - Total pressure at HPC outlet [psia]",
        "s8": "Nf - Physical fan speed [rpm]",
        "s9": "Nc - Physical core speed [rpm]",
        "s10": "epr - Engine pressure ratio (P50/P2)",
        "s11": "Ps30 - Static pressure at HPC outlet [psia]",
        "s12": "Phi - Ratio of fuel flow to Ps30 [pps/psi]",
        "s13": "NRf - Corrected fan speed [rpm]",
        "s14": "NRc - Corrected core speed [rpm]",
        "s15": "BPR - Bypass Ratio",
        "s16": "farB - Burner fuel-air ratio",
        "s17": "htBleed - Bleed Enthalpy",
        "s18": "Nf_dmd - Demanded fan speed [rpm]",
        "s19": "PCNfR_dmd - Demanded corrected fan speed [rpm]",
        "s20": "W31 - HPT coolant bleed [lbm/s]",
        "s21": "W32 - LPT coolant bleed [lbm/s]"
    }

    COLUMN_NAMES: List[str] = ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [
        f"s{i}" for i in range(1, 22)
    ]

    # Standard 7 constant/uninformative sensors across NASA FD001 literature (zero/near-zero variance)
    DEAD_SENSORS: List[str] = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]

    def __init__(self, data_dir: str = None, max_rul_clip: int = 125, rolling_window: int = 5):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        self.data_dir = data_dir
        self.max_rul_clip = max_rul_clip
        self.rolling_window = rolling_window
        self.raw_df = None
        self.processed_df = None
        self.active_sensors = []
        self.feature_cols = []

    def load_raw_data(self) -> pd.DataFrame:
        """Loads authentic NASA C-MAPSS train_FD001.txt."""
        file_path = os.path.join(self.data_dir, "train_FD001.txt")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"NASA C-MAPSS dataset not found at {file_path}")

        df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
        df = df.dropna(axis=1, how="all")
        df.columns = self.COLUMN_NAMES
        self.raw_df = df
        return self.raw_df

    def engineer_features(self) -> pd.DataFrame:
        """
        Computes piece-wise linear clipped RUL target, removes zero-variance constant sensors,
        and engineers rolling degradation trend indicators.
        """
        if self.raw_df is None:
            self.load_raw_data()

        df = self.raw_df.copy()

        # 1. Target Engineering: Ground Truth RUL with Piece-Wise Linear Clipping
        max_cycle_df = df.groupby("engine_id")["cycle"].max().rename("max_cycle")
        df = df.merge(max_cycle_df, on="engine_id")
        df["rul_raw"] = df["max_cycle"] - df["cycle"]
        df["rul_clipped"] = df["rul_raw"].clip(upper=self.max_rul_clip)
        df.drop(columns=["max_cycle"], inplace=True)

        # 2. Filter 21 Sensors: Identify 14 Active Degradation Channels
        sensor_cols = [f"s{i}" for i in range(1, 22)]
        self.active_sensors = [s for s in sensor_cols if s not in self.DEAD_SENSORS]

        # 3. Rolling Statistics for Temporal Degradation Trajectory
        self.feature_cols = []
        for s in self.active_sensors:
            self.feature_cols.append(s)
            
            # Rolling Mean: Smooth out high-frequency measurement noise
            roll_mean = df.groupby("engine_id")[s].transform(
                lambda x: x.rolling(self.rolling_window, min_periods=1).mean()
            )
            col_mean = f"{s}_roll_mean"
            df[col_mean] = roll_mean
            self.feature_cols.append(col_mean)

            # Rolling Std: Quantify vibroacoustic/thermodynamic degradation volatility
            roll_std = df.groupby("engine_id")[s].transform(
                lambda x: x.rolling(self.rolling_window, min_periods=1).std()
            ).fillna(0.0)
            col_std = f"{s}_roll_std"
            df[col_std] = roll_std
            self.feature_cols.append(col_std)

        self.processed_df = df
        return self.processed_df

    def get_train_test_split(
        self, test_size: float = 0.20, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Executes an engine-level group train-test split to prevent temporal leakage.
        Ensures entire engine operational trajectories are quarantined into either train or test.
        """
        if self.processed_df is None:
            self.engineer_features()

        unique_engines = self.processed_df["engine_id"].unique()
        train_engines, test_engines = train_test_split(
            unique_engines, test_size=test_size, random_state=random_state
        )

        train_df = self.processed_df[self.processed_df["engine_id"].isin(train_engines)]
        test_df = self.processed_df[self.processed_df["engine_id"].isin(test_engines)]

        X_train = train_df[self.feature_cols]
        y_train = train_df["rul_clipped"]
        X_test = test_df[self.feature_cols]
        y_test = test_df["rul_clipped"]

        return X_train, X_test, y_train, y_test
