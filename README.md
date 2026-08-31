# 🚀 Turbofan Predictive Maintenance & RUL Engine
> **Gradient Boosted Remaining Useful Life (RUL) Regressors, Piece-Wise Linear Clipping, and Fleet OpEx Optimization**  
> *Industrial IoT · GBRT · NASA C-MAPSS FD001 Benchmark · Piece-Wise Linear Target Clipping · Condition-Based Maintenance*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/SurajChouhan14/Industry-4.0-Predictive-Maintenance-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/SurajChouhan14/Industry-4.0-Predictive-Maintenance-Engine/actions)
[![Benchmark](https://img.shields.io/badge/benchmark-NASA%20C--MAPSS%20FD001-blue.svg)](https://data.nasa.gov/)
[![Tests](https://img.shields.io/badge/tests-4%20passed-brightgreen.svg)]()

---

## 🎯 Executive Overview & Degradation Modeling
An Industrial IoT predictive maintenance platform estimating Remaining Useful Life (RUL) across 21 turbofan telemetry channels from the official NASA C-MAPSS dataset. Implements piece-wise linear RUL clipping and simulates fleet maintenance OpEx savings over reactive run-to-failure and periodic overhaul paradigms.

```
                   Turbofan IoT Sensor Telemetry (21 Channels)
                                       │
         ┌─────────────────────────────┴─────────────────────────────┐
         ▼                                                           ▼
  Rolling Feature Extraction                                Piece-Wise Linear Clipping
   • 5-Cycle Rolling Mean & Std                              • RUL_t = min(125, T_fail - t)
   • Vibration RMS, Temp, Hydraulic Pressure, Acoustic       • Healthy cycle plateau guard
         │                                                           │
         └─────────────────────────────┬─────────────────────────────┘
                                       ▼
                    Gradient Boosted RUL Regressor
                                       │
                                       ▼
                    Fleet Maintenance OpEx Optimizer
                     [ PdM Interventions @ RUL <= 25 ]
```

### 1. Sensor Degradation Feature Pipeline
* Ingests 21 sensor telemetry channels across $20,631$ operational cycles ($100$ turbofan engines).
* Extracts 5-cycle rolling temporal moments (rolling mean, rolling standard deviation) to capture degradation dynamics.

### 2. Piece-Wise Linear RUL Target Clipping
$$\text{Target RUL}_t = \min(RUL_{\max}, \; T_{\text{failure}} - t), \quad RUL_{\max} = 125 \text{ cycles}$$
* Prevents regressors from attempting to learn non-existent degradation patterns during healthy early operating cycles, conforming to standard Prognostics and Health Management (PHM) benchmark methodology.

---

## 📊 Benchmark Performance & Fleet OpEx Economic Simulation

### Holdout Validation Performance ($N = 20$ Out-of-Sample Test Engines / 4,070 Cycles)

| Metric / Economic Parameter | Benchmark Specification / Baseline | Measured Pipeline Performance | Improvement / Status |
|---|:---:|:---:|:---:|
| **Mean Absolute Error (MAE)** | PHM Benchmark Target | **$14.96\text{ operational cycles}$** | **Exact Verified Baseline** |
| **Root Mean Squared Error (RMSE)** | Baseline Regressor | **$20.00\text{ operational cycles}$** | High Degradation Precision |
| **Holdout $R^2$ Score** | Baseline Linear Fit | **$0.7701$** | Strong Telemetry Correlation |
| **Reactive Run-to-Failure Baseline** | Full Fleet Breakdown ($N=20$) | $\$240,000.00$ | $100\%\text{ Failure Risk}$ |
| **Fixed Periodic Overhaul Baseline** | Overhaul Every 60 Cycles | $\$167,500.00$ | Premature Interventions |
| **Predictive Maintenance (PdM) Cost** | Condition-Based Trigger ($RUL \le 25$) | **$\$24,000.00$** | Optimal JIT Maintenance |
| **Fleet OpEx Reduction vs Periodic** | Fixed Schedule Cost | **$-85.67\%$** | **$\$143,500\text{ Net Savings}$** |
| **Fleet OpEx Reduction vs Reactive** | Breakdown Downtime Cost | **$-90.00\%$** | **$\$216,000\text{ Net Savings}$** |

---

## 📁 Repository Structure

```text
Industry-4.0-Predictive-Maintenance-Engine/
├── .github/
│   └── workflows/
│       └── ci.yml                      # Automated CI test & validation workflow
├── .gitignore                          # Git exclusions (pycache, logs)
├── README.md                           # Documentation & degradation modeling
├── Turbofan_Predictive_Maintenance.ipynb # Interactive evaluation & visualization notebook
├── data/
│   ├── industrial_sensor_telemetry.csv # Exported processed telemetry stream
│   └── train_FD001.txt                 # NASA C-MAPSS FD001 run-to-failure benchmark dataset
├── requirements.txt                    # Production dependencies
├── run_pipeline.py                     # 3-phase predictive maintenance pipeline
├── src/
│   ├── __init__.py                     # Package init
│   ├── maintenance_cost_optimizer.py   # Fleet OpEx economic optimization engine
│   ├── rul_regressor.py                # Gradient boosting RUL regression engine
│   └── sensor_telemetry_loader.py      # NASA C-MAPSS dataset ingestion & feature engineering
└── test_predictive_maintenance.py      # 4 automated unit & regression tests
```

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/SurajChouhan14/Industry-4.0-Predictive-Maintenance-Engine.git
cd Industry-4.0-Predictive-Maintenance-Engine
pip install -r requirements.txt
```

### 2. Run Pipeline Benchmark
```bash
python run_pipeline.py
```

### 3. Run Test Suite
```bash
python test_predictive_maintenance.py
```
