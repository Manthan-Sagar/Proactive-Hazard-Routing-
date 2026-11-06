# 📊 Experimental Roadmap & Validation Protocol

## 1. Experimental Objectives

1. **Eliminate Chokepoint Entrapment:** Reduce the entrapment rate from **45.3%** (baseline greedy avoidance) to **< 5%** across 10 random seeds under corridor fire conditions (Origin A).
2. **Preserve Egress Efficiency:** Maintain comparable total evacuation times for unaffected agents without inducing unnecessary mass panic reroutes.
3. **Verify Demographic Harmony:** Preserve priority yielding for elderly and children without deadlock.
4. **Benchmark Computational Feasibility:** Ensure the proactive replanning or policy inference runs in real-time ($< 50\text{ ms}$ per tick for 150 agents).

---

## 2. Experimental Scenarios (Matching Paper 2 Protocol)

All evaluations use the standardized floor plan (`photos/o2_/g.png` and `xml files/02/g.xml`) with 150 agents, age distribution ratio $(6:3:1)$, and 10 seeds ($0 \dots 9$):

| Scenario ID | Origin Description | Coordinates $(X, Y)$ | Baseline Entrapment | Target Entrapment |
| :--- | :--- | :---: | :---: | :---: |
| **Origin A (Primary Challenge)** | Central Corridor Chokepoint | $(181, 350)$ | **45.3% (67.9 trapped)** | **< 5.0% (< 8 trapped)** |
| **Origin B (Control)** | Peripheral Office Room | $(90, 175)$ | 0.0% (0.0 trapped) | 0.0% (0.0 trapped) |
| **Origin C (Secondary Challenge)** | Near-Exit Off-Corridor | $(270, 525)$ | 1.9% (2.9 trapped) | < 1.0% |

---

## 3. Comparative Evaluation Metrics

Each proposed method will be compared against the baseline on five quantitative metrics:

1. **Evacuation Fraction ($\eta_{\text{evac}}$):**
   $$\eta_{\text{evac}} = \frac{N_{\text{evacuated}}}{N_{\text{total}}} \in [0.0, 1.0]$$
2. **Mean Occupants Trapped ($N_{\text{trapped}}$):**
   $$N_{\text{trapped}} = N_{\text{total}} - N_{\text{evacuated}}$$
3. **Avoidance Fraction ($\eta_{\text{avoid}}$):**
   Fraction of the crowd that successfully triggered and executed evasive rerouting.
4. **Total Evacuation Time ($T_{\text{evac}}$):**
   Number of simulation ticks until full crowd clearance (or cap at 4,000 ticks).
5. **Compute Runtime per Tick ($\bar{t}_{\text{step}}$):**
   Wall-clock latency in milliseconds per simulation step.

---

## 4. Phased Step-by-Step Implementation

```
  Phase 1: Architecture & Groundwork (Completed)
     │
     ▼
  Phase 2: Baseline Reproduction Testbed
     • Dedicated test harness isolating Origin A Chokepoint
     • Exact reproduction of the 45% entrapment rate
     │
     ▼
  Phase 3: Track A Implementation (Dynamic Planning)
     • Proactive Lookahead Horizon (H waypoints)
     • Hazard Repulsion Potential Field
     • Event-driven Dynamic Dijkstra / D* Lite Replanning
     • Exit & Corridor Switching
     │
     ▼
  Phase 4: Track B Scaffolding (MARL Foundation)
     • PettingZoo / Gymnasium Environment API
     • Local Feature Extraction Engine
     • Training Loop with Action Masking
     │
     ▼
  Phase 5: Comparative Benchmarks & Visualization
     • Multi-seed statistical verification (Mann-Whitney U)
     • Streamlit Interactive Viewer integration
```

