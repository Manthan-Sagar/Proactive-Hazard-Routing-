# 🛡️ Proactive Hazard Routing for Constraint-Aware Indoor Evacuation

## 📌 Context & Motivation

In multi-agent indoor evacuation simulations, purely geometric shortest paths fail when dynamic, life-threatening hazards (e.g., stochastic spreading fire, smoke accumulation, or structural debris) enter the spatial graph. 

In the baseline study (*"A Demographic-Aware Multi-Agent Crowd Simulation Engine for Hazard-Adaptive Indoor Evacuation"*), the system deployed **decentralized greedy local avoidance**:
* When an occupant's immediate next planned waypoint is burning, the agent queries its immediate 8-neighborhood for an unburned cell minimizing static geodesic distance to the exit.
* **The Failure Mode (Experiment 2b):** At a corridor chokepoint (Origin A), this greedy 1-step reactive heuristic causes a **~45% entrapment rate** (mean 67.9 out of 150 agents fail to escape). Occupants swarm down the corridor toward an exit that is severed downstream, creating dense crowd deadlocks at the flame boundary with zero capacity to turn around.

### Objective
Design, implement, and benchmark **Proactive Hazard Routing** mechanisms that eliminate dead-end entrapment by predicting or sensing hazards ahead of time and executing dynamic rerouting.

---

## 🗺️ Dual-Track Solution Architecture

```
                                 [ Floor Plan & Semantic Graph ]
                                                │
                     ┌──────────────────────────┴──────────────────────────┐
                     ▼                                                     ▼
        [ Track A: Dynamic Planning ]                         [ Track B: Multi-Agent RL ]
        Deterministic & Incremental                           Decentralized Learned Policy
                     │                                                     │
       • Proactive Sensing Horizon                           • Local Multi-Channel Spatial Window
       • Dynamic Path Invalidation                           • Demographic Conditioning (Adult/Child/Elderly)
       • Incremental D* Lite / Replanning                    • MARL Policy (MAPPO / Independent PPO)
       • Voronoi / Topological Exit Switching                • Cooperative Yielding & Bottleneck Flow
                     │                                                     │
                     └──────────────────────────┬──────────────────────────┘
                                                ▼
                             [ Standardized Evaluation Benchmarks ]
                             • Entrapment Rate (% Trapped)
                             • Total Evacuation Time (Ticks)
                             • Bottleneck Specific Throughput (agents/tick)
                             • Per-Tick Computational Overhead (ms)
```

---

## 📑 Documentation Structure

Detailed engineering blueprints and theoretical formulations are documented in [`docs/`](./docs/):

1. **[`01_problem_formulation.md`](./docs/01_problem_formulation.md)**: Mathematical analysis of the chokepoint entrapment failure mode, gradient inversion, and baseline mechanics.
2. **[`02_dynamic_planning_framework.md`](./docs/02_dynamic_planning_framework.md)**: Specification for deterministic proactive planning: Sensing Horizons, Dynamic Dijkstra Invalidation, and $D^*$ Lite.
3. **[`03_marl_architecture.md`](./docs/03_marl_architecture.md)**: Dec-POMDP specification for Multi-Agent Reinforcement Learning, state/action spaces, reward shaping, and training strategy.
4. **[`04_experimental_roadmap.md`](./docs/04_experimental_roadmap.md)**: Phased implementation milestones, validation protocol, and hypothesis testing.

---

## 🚀 Phased Roadmap

| Phase | Description | Deliverables | Status |
| :---: | :--- | :--- | :---: |
| **Phase 1** | **Foundations & Architecture Specification** | Mathematical formulation, project documentation, reproducible baseline testbed | 🟢 Completed |
| **Phase 2** | **Proactive Sensing & Dynamic Incremental Search** | Lookahead horizon $H$, downstream route verification, dynamic exit switching, $D^*$ Lite | 🟡 Next Up |
| **Phase 3** | **MARL Environment & Policy Modeling** | Gymnasium/PettingZoo wrapper, feature extraction, policy training & simulation | ⚪ Planned |
| **Phase 4** | **Comparative Benchmarks & Paper Integration** | Direct side-by-side evaluation against greedy baseline across seeds and fire origins | ⚪ Planned |
