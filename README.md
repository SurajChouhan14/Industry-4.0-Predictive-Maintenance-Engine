# 🚀 Industry 4.0 Sensor Predictive Maintenance Engine
> **Cost-Sensitive Gradient Boosted Machine Failure Classification, Thermodynamic Dissipation Modeling, and Fleet OpEx Loss Optimization**  
> *Industrial IoT · AI4I 2020 Benchmark · Imbalanced Learning · Thermal & Torque Dissipation · Condition-Based Maintenance (PdM)*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/SurajChouhan14/Industry-4.0-Predictive-Maintenance-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/SurajChouhan14/Industry-4.0-Predictive-Maintenance-Engine/actions)
[![Benchmark](https://img.shields.io/badge/benchmark-UCI%20AI4I%202020-blue.svg)](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
[![Tests](https://img.shields.io/badge/tests-4%20passed-brightgreen.svg)]()

---

## 🎯 Executive Overview & Engineering Formulation
An enterprise Industrial IoT predictive maintenance platform detecting and classifying catastrophic machine tool breakdowns across 10,000 sensor telemetry records from the authentic **UCI AI4I 2020 Predictive Maintenance Dataset**. Implements domain-specific thermodynamic and mechanical power dissipation feature engineering to overcome severe **3.39% operational failure class imbalance** (339 recorded breakdowns across 5 distinct failure modes).

```
                      Industrial Sensor Telemetry Stream (10,000 Records)
                                          │
            ┌─────────────────────────────┴─────────────────────────────┐
            ▼                                                           ▼
    Thermodynamic Dissipation                                 Mechanical Power Dissipation
   • Delta T = T_process - T_air                             • Power (kW) = Torque * 2*pi*RPM / 60
   • Temp Ratio = T_process / T_air                          • Overstrain = Torque * Tool Wear
            │                                                           │
            └─────────────────────────────┬─────────────────────────────┘
                                          ▼
                       Cost-Sensitive Gradient Boosting
                     [ ROC-AUC: 96.03% | PR-AUC: 89.82% ]
                                          │
                                          ▼
                      Fleet Maintenance OpEx Loss Optimizer
                        [ >85% Fleet Maintenance Savings ]
```

---

## 🔬 Domain Physics & Feature Engineering

### 1. Thermal Dissipation ($\Delta T$)
$$\Delta T = T_{	ext{process}} - T_{	ext{air}} \quad [	ext{Kelvin}]$$
* Heat Dissipation Failure (HDF) occurs when convective heat dissipation is insufficient ($\Delta T < 8.6	ext{ K}$) during low-speed high-torque operation ($	ext{Speed} < 1380	ext{ rpm}$).

### 2. Mechanical Power Dissipation ($P$)
$$P = 	au \cdot \omega = 	au \cdot \left( rac{2\pi \cdot 	ext{RPM}}{60} ight) \cdot 10^{-3} \quad [	ext{kW}]$$
* Power Failure (PWF) occurs when required cutting power exceeds motor electrical rating ($P > 9.0	ext{ kW}$) or drops below minimum operational threshold ($P < 3.5	ext{ kW}$).

### 3. Tool Overstrain Metric ($S$)
$$S = 	au \cdot t_{	ext{wear}} \quad [	ext{N}\cdot	ext{m}\cdot	ext{min}]$$
* Overstrain Failure (OSF) occurs when the cumulative product of cutting torque and tool wear exceeds metallurgical stress limits ($11,000$ for Type L, $12,000$ for Type M, $13,000$ for Type H).

---

## 📊 Benchmark Performance & Economic Loss Reduction

### Out-of-Sample Test Set Performance ($N = 2,000$ Industrial Machines / Stratified 80/20 Split)

| Evaluation Metric / Parameter | Baseline Target / Specification | Measured Pipeline Performance | Status / Significance |
|---|:---:|:---:|:---:|
| **Total Sensor Records Ingested** | UCI AI4I 2020 Benchmark | **$10,000	ext{ Records}$** | Exact Dataset Match |
| **Operational Failure Rate** | Imbalanced Benchmark | **$3.39\%	ext{ (339 Failures)}$** | Exact Real Imbalance |
| **Out-of-Sample ROC-AUC** | Industrial Benchmark ($\ge 95\%$) | **$96.03\%$** | High Discrimination |
| **Precision-Recall AUC (PR-AUC)** | Class Imbalance Baseline ($\ge 80\%$) | **$89.82\%$** | **Defends Resume 85.2%** |
| **Reactive 'Run-to-Failure' Cost** | $N=2,000$ Fleet ($C_{	ext{fail}}=\$10	ext{k}$) | $\$680,000.00$ | $100\%$ Catastrophic Risk |
| **Fixed Periodic Overhaul Cost** | Planned Overhaul ($C_{	ext{plan}}=\$500$) | $\$1,000,000.00$ | Over-Maintenance Waste |
| **Predictive Maintenance (PdM) Cost** | Condition-Based Threshold ($T^*$) | **$\$89,600.00$** | **Optimal Economic Policy** |
| **OpEx Cost Reduction vs Reactive** | Catastrophic Breakdown Losses | **$-86.82\%$** | **$\$590,400	ext{ Net Savings}$** |
| **OpEx Cost Reduction vs Periodic** | Unnecessary Overhaul Losses | **$-91.04\%$** | **$\$910,400	ext{ Net Savings}$** |

---

## 📁 Repository Structure

```text
Industry-4.0-Predictive-Maintenance-Engine/
├── .github/
│   └── workflows/
│       └── ci.yml                     # Automated GitHub Actions CI workflow
├── .gitignore                         # Git exclusions
├── README.md                          # Documentation & physics derivations
├── data/
│   └── ai4i2020.csv                   # Authentic UCI AI4I 2020 dataset (10,000 records)
├── requirements.txt                   # Production dependencies
├── run_pipeline.py                    # 3-stage end-to-end execution pipeline
├── src/
│   ├── __init__.py                    # Package init
│   ├── ai4i_data_loader.py            # AI4I ingestion & thermodynamic feature engineering
│   ├── failure_classifier.py          # Cost-sensitive gradient boosted failure classifier
│   └── maintenance_cost_optimizer.py  # Fleet maintenance loss matrix & OpEx optimizer
└── test_predictive_maintenance.py     # Automated unit & integration tests
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

### 3. Run Automated Tests
```bash
python test_predictive_maintenance.py
```
