"""
baseline_evaluator.py
=====================
Benchmark harness reproducing the baseline greedy-avoidance performance
under corridor chokepoint fire conditions (Experiment 2b / Origin A in Paper 2).
"""

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image

# Add root and paper directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
parent_root = os.path.dirname(project_root)

sys.path.insert(0, os.path.join(parent_root, 'paper'))
import app as paper_app
import simulation as sim
import reviewer_experiments as rx
from src.config import SimulationConfig, BenchmarkLocations


def load_ground_floor():
    """Loads Ground Floor raster and XML annotations from repository assets."""
    img_path = os.path.join(parent_root, 'photos', 'o2_', 'g.png')
    xml_path = os.path.join(parent_root, 'xml files', '02', 'g.xml')

    assert os.path.exists(img_path), f"Missing floor image at {img_path}"
    assert os.path.exists(xml_path), f"Missing XML annotations at {xml_path}"

    img = np.array(Image.open(img_path).convert('RGB'))
    with open(xml_path, 'r', encoding='utf-8') as f:
        xml_content = f.read()

    annotations = paper_app.parse_xml_complete(xml_content)
    free_space = paper_app.detect_free_space(img, annotations)
    graph = paper_app.build_graph_with_zones(free_space, annotations, [], [], 'Ground', False, False)
    exit_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'exit']

    return img, free_space, graph, exit_nodes, annotations


def evaluate_baseline_greedy(n_seeds: int = 5, base_seed: int = 0):
    """
    Executes seeded runs of the baseline greedy avoidance engine
    on Origin A (Corridor Chokepoint) to establish the ground truth entrapment rate.
    """
    print(f"Loading Ground Floor substrate...")
    _, free_space, graph, exits, _ = load_ground_floor()
    cfg = SimulationConfig()
    loc = BenchmarkLocations()

    fire_origin = loc.origin_a_chokepoint
    print(f"Executing baseline greedy evaluation across {n_seeds} seeds at Origin A {fire_origin}...")

    results = []
    for s in range(base_seed, base_seed + n_seeds):
        rx._seed(s)
        hist, total_ticks, avoid_frac = sim.run_headless_experiment_v2(
            graph, free_space, exits, cfg.num_agents, cfg.age_ratio,
            fire_mode='dynamic', fire_start=fire_origin,
            fire_spread_chance=cfg.fire_spread_chance, fire_interval=cfg.fire_interval
        )
        evac_count = hist[-1] if hist else 0
        evac_frac = evac_count / cfg.num_agents
        trapped = cfg.num_agents - evac_count
        results.append({
            "seed": s,
            "evac_count": evac_count,
            "evac_fraction": evac_frac,
            "trapped_count": trapped,
            "trapped_pct": (trapped / cfg.num_agents) * 100.0,
            "avoidance_fraction": avoid_frac,
            "total_ticks": total_ticks
        })
        print(f"  [Seed {s}] Evacuated: {evac_count}/{cfg.num_agents} ({evac_frac*100:.1f}%), Trapped: {trapped} ({(trapped/cfg.num_agents)*100:.1f}%), Avoidance: {avoid_frac:.2f}, Ticks: {total_ticks}")

    df = pd.DataFrame(results)
    mean_evac = df["evac_fraction"].mean()
    mean_trapped = df["trapped_count"].mean()
    mean_trapped_pct = df["trapped_pct"].mean()
    mean_avoid = df["avoidance_fraction"].mean()

    print("\n=== BASELINE GREEDY SUMMARY (ORIGIN A CHOKEPOINT) ===")
    print(f"Mean Evacuated Fraction : {mean_evac:.3f} ± {df['evac_fraction'].std():.3f}")
    print(f"Mean Occupants Trapped  : {mean_trapped:.1f} / {cfg.num_agents} ({mean_trapped_pct:.1f}%)")
    print(f"Mean Avoidance Fraction : {mean_avoid:.3f}")
    print("====================================================\n")
    return df


if __name__ == "__main__":
    evaluate_baseline_greedy(n_seeds=3, base_seed=0)
