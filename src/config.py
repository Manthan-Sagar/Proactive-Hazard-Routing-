"""
config.py
=========
Global configuration and parameter specifications for proactive hazard routing,
simulation engine, and benchmark evaluation.
"""

from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class SimulationConfig:
    # Crowd parameters
    num_agents: int = 150
    age_ratio: Tuple[int, int, int] = (6, 3, 1)  # adult : young : elderly
    step_size: int = 2                            # grid step size (pixels)
    max_ticks: int = 4000                         # timeout cap

    # Hazard parameters
    fire_spread_chance: float = 0.08              # sigma: probability of spreading per check
    fire_interval: int = 5                        # ticks between fire spread events

    # Demographic speeds (ticks per step)
    ticks_adult: Tuple[int, int] = (2, 3)
    ticks_young: Tuple[int, int] = (3, 4)
    ticks_elderly: Tuple[int, int] = (4, 6)

    # Demographic yielding priorities (lower number = higher yielding priority)
    priority_young: int = 1
    priority_elderly: int = 2
    priority_adult: int = 3


@dataclass(frozen=True)
class ProactiveRoutingConfig:
    # Track A: Lookahead & Proactive Sensing
    sensing_radius: float = 40.0                  # Euclidean sensing radius (pixels)
    lookahead_waypoints: int = 25                 # Number of future path waypoints to verify
    hazard_buffer_margin: float = 15.0            # Repulsion buffer distance around flames (pixels)
    hazard_cost_multiplier: float = 10.0          # Max penalty scalar for proximity to fire
    hazard_decay_lambda: float = 15.0             # Exponential decay rate of hazard potential

    # Dynamic replanning trigger
    enable_exit_switching: bool = True            # Allow switching to alternate exit / stairs
    min_reroute_interval: int = 10                # Minimum ticks between global replans per agent


@dataclass(frozen=True)
class BenchmarkLocations:
    # Benchmark Origins on Ground Floor (measured from Dijkstra traversal centrality)
    origin_a_chokepoint: Tuple[int, int] = (250, 580)     # Primary exit bottleneck corridor (cuts off main exit)
    origin_b_peripheral: Tuple[int, int] = (90, 175)      # Peripheral room (0% entrapment)
    origin_c_offcorridor: Tuple[int, int] = (270, 525)    # Near exit off-corridor (<2% entrapment)

    # Convenience aliases and alternative names
    origin_a: Tuple[int, int] = (250, 580)
    origin_b: Tuple[int, int] = (90, 175)
    origin_b_west: Tuple[int, int] = (90, 175)
    origin_c: Tuple[int, int] = (270, 525)

