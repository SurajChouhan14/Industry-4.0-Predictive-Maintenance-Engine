# 🚀 Turbofan Predictive Maintenance & RUL Engine
### NASA C-MAPSS FD001 Dataset | Gradient Boosted RUL Regressors | Piece-Wise Linear Clipping | OpEx Simulation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Industrial IoT](https://img.shields.io/badge/Industrial%20IoT-NASA%20C--MAPSS-success.svg)](https://data.nasa.gov/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An Industrial IoT predictive maintenance engine estimating Remaining Useful Life (RUL) across 21 turbofan telemetry channels from the official NASA C-MAPSS dataset. Implements piece-wise linear RUL clipping and simulates fleet maintenance OpEx savings.

---

## 📌 Methodology & Degradation Modeling

### 1. Sensor Degradation Engineering:
Extracts rolling statistical moments (mean, standard deviation, trends) across 21 sensor channels.

### 2. Piece-Wise Linear RUL Target Clipping:
$$\text{Target RUL}_t = \min(RUL_{\max}, \; T_{\text{failure}} - t), \quad RUL_{\max} = 125 \text{ cycles}$$
* Prevents regressors from learning non-existent degradation patterns during healthy early operating cycles.

---

## 📊 Benchmark Performance & OpEx Economic Simulation
* **Dataset:** NASA C-MAPSS FD001 ($20,631$ cycles across 100 jet engine units).
* **Holdout Validation Performance ($N = 20$ Test Engines / 4,070 Cycles):**
  * **Mean Absolute Error (MAE):** $\mathbf{14.96 \text{ operational cycles}}$
  * **Root Mean Squared Error (RMSE):** $20.00\text{ cycles}$
  * **Holdout $R^2$:** $0.7701$
* **Fleet OpEx Economic Simulation:**
  * Run-to-Failure Cost Baseline: $\$240,000.00$
  * Periodic Overhaul Baseline: $\$167,500.00$
  * Predictive Maintenance Fleet Cost: $\$24,000.00$
  * **Net Fleet Maintenance OpEx Reduction:** **-85.67% capital savings** over periodic overhauls.

---

## 📂 Repository Structure
```
Industry-4.0-Predictive-Maintenance-Engine/
├── src/
│   ├── predictive_maintenance_engine.py # Feature pipeline & GBRT RUL regressor
│   └── data_loader.py              # NASA C-MAPSS dataset ingestion
├── Turbofan_Predictive_Maintenance.ipynb # Interactive evaluation notebook
├── run_pipeline.py                 # Pipeline execution script
├── test_predictive_maintenance.py  # Unit testing suite (4/4 passing)
└── requirements.txt                # Production dependencies
```

---

## 🚀 Quickstart & Reproducibility
```bash
git clone https://github.com/SurajChouhan14/Industry-4.0-Predictive-Maintenance-Engine.git
cd Industry-4.0-Predictive-Maintenance-Engine
pip install -r requirements.txt
python run_pipeline.py
python -m unittest test_predictive_maintenance.py
```
