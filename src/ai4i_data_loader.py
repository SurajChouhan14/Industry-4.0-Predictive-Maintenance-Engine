"""
AI4I 2020 Predictive Maintenance Data Loader & Physics Feature Engineering Module.
Authentic UCI AI4I 2020 Dataset Ingestion & Thermodynamic / Power Dissipation Pipelines.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any

class AI4IDataLoader:
    """
    Ingests and transforms authentic UCI AI4I 2020 Predictive Maintenance dataset (10,000 records).
    Engineers thermodynamic temperature differentials, mechanical power dissipation,
    and tool overstrain interaction features.
    """

    FAILURE_MODES = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

    def __init__(self, data_path: str = None):
        if data_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(base_dir, 'data', 'ai4i2020.csv')
        self.data_path = data_path
        self.raw_df = None
        self.processed_df = None
        self.feature_names = []

    def load_data(self) -> pd.DataFrame:
        """Loads raw AI4I 2020 CSV dataset and validates schema integrity."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"AI4I dataset not found at: {self.data_path}")
        
        self.raw_df = pd.read_csv(self.data_path)
        
        # Verify schema expectations
        expected_cols = [
            'UDI', 'Product ID', 'Type', 'Air temperature [K]',
            'Process temperature [K]', 'Rotational speed [rpm]',
            'Torque [Nm]', 'Tool wear [min]', 'Machine failure'
        ]
        for col in expected_cols:
            if col not in self.raw_df.columns:
                raise ValueError(f"Missing required column in AI4I dataset: {col}")
                
        return self.raw_df

    def engineer_features(self) -> pd.DataFrame:
        """
        Engineers physics-informed sensor degradation signals:
        1. Thermal Dissipation (Delta T = Process Temp - Air Temp)
        2. Mechanical Power Dissipation (Power = Torque * Angular Velocity in kW)
        3. Tool Overstrain Metric (Strain = Torque * Tool Wear)
        4. Thermodynamic Temperature Ratio (Process Temp / Air Temp)
        5. Rotational Torque Kinematic Ratio (RPM / Torque)
        """
        if self.raw_df is None:
            self.load_data()

        df = self.raw_df.copy()

        # 1. Thermal Dissipation (K) - Heat Dissipation Failure (HDF) indicator
        df['thermal_dissipation_K'] = df['Process temperature [K]'] - df['Air temperature [K]']

        # 2. Power Dissipation (kW) - Power Failure (PWF) indicator (P = Torque * 2*pi*rpm / 60)
        df['power_dissipation_kW'] = (df['Torque [Nm]'] * df['Rotational speed [rpm]'] * (2 * np.pi / 60.0)) / 1000.0

        # 3. Tool Overstrain (min*Nm) - Overstrain Failure (OSF) indicator
        df['overstrain_torque_wear'] = df['Torque [Nm]'] * df['Tool wear [min]']

        # 4. Temperature Ratio
        df['temp_ratio'] = df['Process temperature [K]'] / df['Air temperature [K]']

        # 5. Kinematic Torque-to-Speed Ratio
        df['rot_torque_ratio'] = df['Rotational speed [rpm]'] / (df['Torque [Nm]'] + 1e-6)

        # 6. Type Encoding (L: Low variant 50%, M: Medium variant 30%, H: High variant 20%)
        df['type_encoded'] = df['Type'].map({'L': 0, 'M': 1, 'H': 2}).fillna(0).astype(int)

        # 7. Non-Linear Risk Zone Flags (Domain Physics Rules)
        df['hdf_risk_zone'] = ((df['thermal_dissipation_K'] < 8.6) & (df['Rotational speed [rpm]'] < 1380)).astype(float)
        df['pwf_risk_zone'] = ((df['power_dissipation_kW'] < 3.5) | (df['power_dissipation_kW'] > 9.0)).astype(float)
        
        osf_thresh = np.where(df['Type'] == 'L', 11000, np.where(df['Type'] == 'M', 12000, 13000))
        df['osf_risk_zone'] = (df['overstrain_torque_wear'] > osf_thresh).astype(float)

        self.feature_names = [
            'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
            'Torque [Nm]', 'Tool wear [min]', 'thermal_dissipation_K',
            'power_dissipation_kW', 'overstrain_torque_wear', 'temp_ratio',
            'rot_torque_ratio', 'type_encoded', 'hdf_risk_zone', 'pwf_risk_zone', 'osf_risk_zone'
        ]

        self.processed_df = df
        return self.processed_df

    def get_train_test_split(
        self, test_size: float = 0.20, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Splits processed data into stratified train/test partitions."""
        from sklearn.model_selection import train_test_split

        if self.processed_df is None:
            self.engineer_features()

        X = self.processed_df[self.feature_names]
        y = self.processed_df['Machine failure']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test
