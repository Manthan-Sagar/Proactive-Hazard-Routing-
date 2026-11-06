# 🛡️ Proactive Hazard-Adaptive Indoor Routing (PHAIR)

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Gymnasium 1.4.0](https://img.shields.io/badge/Gymnasium-1.4.0-green.svg)](https://gymnasium.farama.org/)
[![PyTorch 2.15](https://img.shields.io/badge/PyTorch-2.15-orange.svg)](https://pytorch.org/)

A dual-track framework for proactive hazard routing during architectural indoor fires:
1. **Track A (Dynamic Search):** Downstream trajectory verification ($H$), Euclidean distance transform hazard buffer, and event-driven graph repair.
2. **Track B (MARL):** Dec-POMDP multi-agent Gymnasium environment (`HazardEvacEnv`) with local spatial vision tensors (21 × 21 × 6) for multi-agent reinforcement learning.

---

## 📁 Repository Structure

```
proactive_hazard_routing/
├── data/                          # Local isolated CAD rasters & annotations
│   ├── floors/                    # Ground, 1st floor, Penthouse, Office 1 PNGs
│   └── xml/                       # Polyline XML annotations (doors, exits, stairs)
├── docs/
│   ├── 01_problem_formulation.md  # Continuous-to-grid mapping and failure modes
│   ├── 02_dynamic_planning_framework.md # Dynamic planning & lookahead H
│   ├── 03_marl_architecture.md    # Dec-POMDP formulation & reward shaping
│   └── 04_experimental_roadmap.md # Evaluation protocols & seed statistics
├── src/
│   ├── __init__.py
│   ├── config.py                  # Simulation & routing hyperparameters
│   ├── floor_loader.py            # Isolated floor loader & navigation graph builder
│   ├── proactive_engine.py        # Track A: Proactive dynamic routing engine
│   └── env_evac.py                # Track B: Gymnasium multi-agent environment
├── tests/
│   ├── test_env.py                # Gymnasium environment unit tests
│   └── test_proactive_vs_baseline.py # Comparative benchmark test
├── app.py                         # Interactive Streamlit dashboard
├── PROJECT_STATUS.md              # Detailed project milestone tracking
└── README.md
```

---

## 🚀 Quickstart

### 1. Launch the Web Dashboard
```powershell
streamlit run app.py
```

### 2. Run Gymnasium Environment Verification
```powershell
python tests/test_env.py
```

### 3. Run Comparative Egress Benchmark
```powershell
python tests/test_proactive_vs_baseline.py
```
