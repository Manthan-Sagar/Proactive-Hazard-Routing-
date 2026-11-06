"""
test_proactive_vs_baseline.py
==============================
Side-by-side comparative validation testing:
  1. Baseline Greedy Engine (Paper 2)
  2. Proactive Hazard-Adaptive Engine (Track A)
under identical seed and corridor chokepoint conditions.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add proactive_hazard_routing to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
parent_root = os.path.dirname(project_root)

sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(parent_root, 'paper'))

import simulation as sim
import reviewer_experiments as rx
from src.baseline_evaluator import load_ground_floor
from src.proactive_engine import run_proactive_simulation
from src.config import SimulationConfig, BenchmarkLocations


def run_comparative_experiment(n_seeds: int = 3, base_seed: int = 0):
    print("================================================================")
    print("PROACTIVE HAZARD ROUTING VS BASELINE GREEDY COMPARATIVE EXPERIMENT")
    print("================================================================")

    _, free_space, graph, exits, _ = load_ground_floor()
    cfg = SimulationConfig(num_agents=100)  # Use 100 agents for quick benchmark
    loc = BenchmarkLocations()
    fire_origin = loc.origin_a_chokepoint

    records = []
    print(f"Fire Origin: {fire_origin} (Corridor Chokepoint), Crowd: {cfg.num_agents} occupants\n")

    for s in range(base_seed, base_seed + n_seeds):
        # 1. Baseline Run
        rx._seed(s)
        h_base, t_base, avoid_base = sim.run_headless_experiment_v2(
            graph, free_space, exits, cfg.num_agents, cfg.age_ratio,
            fire_mode='dynamic', fire_start=fire_origin,
            fire_spread_chance=cfg.fire_spread_chance, fire_interval=cfg.fire_interval
        )
        evac_base = h_base[-1] if h_base else 0
        trapped_base = cfg.num_agents - evac_base

        # 2. Proactive Run
        rx._seed(s)
        h_pro, t_pro, avoid_pro, evac_pro = run_proactive_simulation(
            graph, free_space, exits, num_agents=cfg.num_agents, age_ratio=cfg.age_ratio,
            fire_start=fire_origin, lookahead=25, sensing_radius=40.0
        )
        trapped_pro = cfg.num_agents - evac_pro

        print(f"[Seed {s}]")
        print(f"  BASELINE  : Evacuated={evac_base}/{cfg.num_agents} ({(evac_base/cfg.num_agents)*100:.1f}%), Trapped={trapped_base}, Ticks={t_base}")
        print(f"  PROACTIVE : Evacuated={evac_pro}/{cfg.num_agents} ({(evac_pro/cfg.num_agents)*100:.1f}%), Trapped={trapped_pro}, Ticks={t_pro}")

        records.append({
            "seed": s,
            "baseline_evac": evac_base,
            "baseline_trapped": trapped_base,
            "baseline_ticks": t_base,
            "proactive_evac": evac_pro,
            "proactive_trapped": trapped_pro,
            "proactive_ticks": t_pro,
        })

    df = pd.DataFrame(records)
    print("\n======================= SUMMARY COMPARISON =======================")
    print(f"BASELINE  -> Mean Evacuated: {df['baseline_evac'].mean():.1f} ({df['baseline_evac'].mean()/cfg.num_agents*100:.1f}%), Mean Trapped: {df['baseline_trapped'].mean():.1f}")
    print(f"PROACTIVE -> Mean Evacuated: {df['proactive_evac'].mean():.1f} ({df['proactive_evac'].mean()/cfg.num_agents*100:.1f}%), Mean Trapped: {df['proactive_trapped'].mean():.1f}")
    print("==================================================================")
    return df


if __name__ == "__main__":
    run_comparative_experiment(n_seeds=2, base_seed=0)
