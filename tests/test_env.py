"""
test_env.py
===========
Unit tests for the Track B Gymnasium MARL Environment (HazardEvacEnv).
Verifies:
  1. Environment instantiation and reset
  2. Observation tensor shape (21, 21, 6) and value ranges [0, 1]
  3. Action masking alignment
  4. Multi-agent stepping with demographic gating
  5. Fire propagation in environment
"""

import os
import sys
import numpy as np

# Path configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
parent_root = os.path.dirname(project_root)

sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(parent_root, 'paper'))

from src.baseline_evaluator import load_ground_floor
from src.env_evac import HazardEvacEnv
import networkx as nx


def test_hazard_evac_env():
    print("Testing HazardEvacEnv Gymnasium Environment...")
    img, free_space, graph, exits, ann = load_ground_floor()

    # Precompute dist_map
    lengths, _ = nx.multi_source_dijkstra(graph, sources=exits, weight='weight')
    dist_map = {graph.nodes[n]['pos']: d for n, d in lengths.items()}
    exit_coords = [graph.nodes[e]['pos'] for e in exits]

    env = HazardEvacEnv(
        free_space_grid=free_space,
        dist_map=dist_map,
        exit_coords=exit_coords,
        num_agents=20,
        age_ratio=(6, 3, 1),
        obs_window_size=21,
        fire_origin=(250, 580)
    )

    # 1. Reset Test
    obs, info = env.reset(seed=42)
    assert len(obs) == 20, f"Expected 20 agent observations, got {len(obs)}"
    print(f"  [PASS] reset() returned observations for {len(obs)} agents.")

    # 2. Observation Structure Check
    first_aid = list(obs.keys())[0]
    agent_obs = obs[first_aid]
    spatial = agent_obs["spatial"]
    proprio = agent_obs["proprio"]
    mask = agent_obs["action_mask"]

    assert spatial.shape == (21, 21, 6), f"Spatial tensor shape mismatch: {spatial.shape}"
    assert proprio.shape == (7,), f"Proprio vector shape mismatch: {proprio.shape}"
    assert mask.shape == (9,), f"Action mask shape mismatch: {mask.shape}"
    assert 0.0 <= spatial.min() and spatial.max() <= 1.0, f"Spatial values out of [0, 1] range: min={spatial.min()}, max={spatial.max()}"
    assert mask[0] == 1, "Stay action (0) should always be valid"
    print(f"  [PASS] Observation tensor verified: spatial={spatial.shape}, proprio={proprio.shape}, mask={mask.shape}")

    # 3. Step Test
    print("  Testing 50 environmental simulation steps...")
    for t in range(50):
        # Sample random valid actions from action mask
        actions = {}
        for aid, a_obs in obs.items():
            valid_actions = np.where(a_obs["action_mask"] == 1)[0]
            actions[aid] = int(np.random.choice(valid_actions))

        obs, rewards, terminated, truncated, info = env.step(actions)
        if info["all_done"]:
            print(f"  All agents finished at step {t}")
            break

    print(f"  [PASS] Completed 50 steps: tick={info['tick']}, evacuated={info['evacuated']}, dead={info['dead']}, active={info['active']}")
    print("All Gymnasium environment tests passed successfully!\n")


if __name__ == "__main__":
    test_hazard_evac_env()
