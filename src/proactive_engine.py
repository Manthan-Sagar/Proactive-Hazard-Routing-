"""
proactive_engine.py
===================
Track A: Proactive Hazard-Adaptive Routing Engine.
Implements:
  1. Downstream Trajectory Verification (Lookahead Horizon H)
  2. Hazard Repulsion Potential Field (Distance Transform Buffer)
  3. Dynamic Event-Driven Path Invalidation & Replanning
  4. Demographic-Aware Bottleneck Yielding
"""

import random
import numpy as np
import networkx as nx
from scipy.ndimage import distance_transform_edt
from typing import List, Tuple, Dict, Set, Optional

from src.config import SimulationConfig, ProactiveRoutingConfig


class ProactiveEvacuationAgent:
    def __init__(self, agent_id: int, start_pos: Tuple[float, float],
                 path: List[Tuple[float, float]], agent_type: str = 'adult',
                 lookahead: int = 25, sensing_radius: float = 40.0):
        self.agent_id = agent_id
        self.current_pos = start_pos
        self.path = path
        self.path_index = 0
        self.is_evacuated = False
        self.tick_counter = 0
        self.agent_type = agent_type

        # Lookahead & Proactive Sensing
        self.lookahead = lookahead
        self.sensing_radius = sensing_radius
        self.entered_avoidance = False
        self.proactive_reroute_count = 0
        self.last_reroute_tick = -100

        # Demographic speeds & priorities matching baseline
        if agent_type == 'young':
            self.ticks_per_step = 3 + np.random.randint(0, 2)
            self.priority = 1
        elif agent_type == 'elderly':
            self.ticks_per_step = 4 + np.random.randint(0, 3)
            self.priority = 2
        else:
            self.ticks_per_step = 2 + np.random.randint(0, 2)
            self.priority = 3

    def get_next_pos(self) -> Tuple[float, float]:
        if self.path_index + 1 < len(self.path):
            return self.path[self.path_index + 1]
        return self.current_pos

    def check_downstream_hazard(self, fire_grid: np.ndarray, buffer_grid: np.ndarray) -> bool:
        """
        Inspects upcoming H waypoints along the planned trajectory.
        Returns True if the trajectory passes through fire or dangerous smoke buffer.
        """
        h, w = fire_grid.shape
        end_idx = min(len(self.path), self.path_index + self.lookahead)
        for i in range(self.path_index + 1, end_idx):
            wx, wy = int(self.path[i][0]), int(self.path[i][1])
            if 0 <= wy < h and 0 <= wx < w:
                if fire_grid[wy, wx] or buffer_grid[wy, wx]:
                    return True
        return False

    def move(self):
        self.path_index += 1
        self.current_pos = self.path[self.path_index]
        if self.path_index >= len(self.path) - 1:
            self.is_evacuated = True


class ProactiveCrowdSimulationEngine:
    def __init__(self, nav_graph: nx.Graph, free_space_grid: np.ndarray, exit_nodes: list,
                 sim_config: Optional[SimulationConfig] = None,
                 routing_config: Optional[ProactiveRoutingConfig] = None,
                 fire_mode: str = 'dynamic', fire_start: Tuple[int, int] = (0, 0)):

        self.sim_cfg = sim_config or SimulationConfig()
        self.rout_cfg = routing_config or ProactiveRoutingConfig()

        self.nav_graph = nav_graph.copy()
        self.free_space_grid = free_space_grid
        self.exit_nodes = exit_nodes
        self.h, self.w = free_space_grid.shape

        self.occupancy_grid = np.zeros_like(free_space_grid, dtype=bool)
        self.fire_grid = np.zeros_like(free_space_grid, dtype=bool)
        self.hazard_buffer_grid = np.zeros_like(free_space_grid, dtype=bool)
        self.fire_front = []

        self.fire_mode = fire_mode
        self.current_tick = 0
        self.evacuated_count = 0
        self.agents: List[ProactiveEvacuationAgent] = []
        self.frame_history = []

        # Multi-source Dijkstra distance maps
        self.dist_map: Dict[Tuple[float, float], float] = {}
        self.paths: Dict[str, List[str]] = {}
        self._compute_global_escape_paths()

        if fire_mode != 'none':
            self.start_fire(int(fire_start[0]), int(fire_start[1]))

    def _compute_global_escape_paths(self):
        """Computes shortest path from all navigable nodes to the nearest exit."""
        active_exits = [e for e in self.exit_nodes if e in self.nav_graph and not self.fire_grid[int(self.nav_graph.nodes[e]['pos'][1]), int(self.nav_graph.nodes[e]['pos'][0])]]
        if not active_exits:
            active_exits = self.exit_nodes

        lengths, paths = nx.multi_source_dijkstra(self.nav_graph, sources=active_exits, weight='weight')
        self.paths = paths
        self.dist_map = {self.nav_graph.nodes[n]['pos']: d for n, d in lengths.items()}

    def start_fire(self, x: int, y: int):
        if 0 <= y < self.h and 0 <= x < self.w:
            self.fire_grid[y, x] = True
            self.fire_front.append((x, y))
            self._update_hazard_buffer()

    def _update_hazard_buffer(self):
        """Calculates repulsion buffer surrounding flames using Euclidean Distance Transform."""
        if not self.fire_front:
            self.hazard_buffer_grid[:] = False
            return
        dist = distance_transform_edt(~self.fire_grid)
        self.hazard_buffer_grid = (dist <= self.rout_cfg.hazard_buffer_margin) & (self.free_space_grid > 0)

    def spread_fire(self):
        new_front = []
        spread_chance = self.sim_cfg.fire_spread_chance
        for fx, fy in self.fire_front:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx_c, ny_c = fx + dx, fy + dy
                if 0 <= ny_c < self.h and 0 <= nx_c < self.w:
                    if self.free_space_grid[ny_c, nx_c] and not self.fire_grid[ny_c, nx_c]:
                        if random.random() < spread_chance:
                            self.fire_grid[ny_c, nx_c] = True
                            new_front.append((nx_c, ny_c))
        self.fire_front.extend(new_front)
        if new_front:
            self._update_hazard_buffer()

    def add_agent(self, agent: ProactiveEvacuationAgent):
        self.agents.append(agent)
        cx, cy = int(agent.current_pos[0]), int(agent.current_pos[1])
        if 0 <= cy < self.h and 0 <= cx < self.w:
            self.occupancy_grid[cy, cx] = True

    def replan_agent(self, agent: ProactiveEvacuationAgent) -> bool:
        """
        Dynamically replans an escape trajectory avoiding burned and buffer zones.
        """
        # Find closest node in nav_graph to current position
        cx, cy = int(agent.current_pos[0]), int(agent.current_pos[1])
        best_node, best_d = None, float('inf')
        for dx in range(-4, 5, 2):
            for dy in range(-4, 5, 2):
                tx, ty = cx + dx, cy + dy
                nid = f"Ground_pix_{tx}_{ty}"
                if nid in self.nav_graph and not self.fire_grid[ty, tx]:
                    d = np.hypot(dx, dy)
                    if d < best_d:
                        best_d, best_node = d, nid

        if not best_node or best_node not in self.paths:
            return False

        # Build trajectory from path dictionary
        path_nodes = self.paths[best_node][::-1]
        coords = [self.nav_graph.nodes[n]['pos'] for n in path_nodes]
        if coords:
            agent.path = coords
            agent.path_index = 0
            agent.proactive_reroute_count += 1
            agent.entered_avoidance = True
            agent.last_reroute_tick = self.current_tick
            return True
        return False

    def step(self):
        self.current_tick += 1

        # Fire spread update
        if self.fire_mode == 'dynamic' and self.current_tick % self.sim_cfg.fire_interval == 0:
            self.spread_fire()
            # If fire entered paths, update global distance field
            self._compute_global_escape_paths()

        active_agents = [a for a in self.agents if not a.is_evacuated]
        # Demographic priority sorting
        active_agents.sort(key=lambda a: (a.priority, random.random()))

        current_frame_positions = []

        for agent in active_agents:
            agent.tick_counter += 1
            if agent.tick_counter >= agent.ticks_per_step:
                agent.tick_counter = 0

                # 1. PROACTIVE CHECK: Is downstream path blocked?
                if (self.current_tick - agent.last_reroute_tick > self.rout_cfg.min_reroute_interval and
                        agent.check_downstream_hazard(self.fire_grid, self.hazard_buffer_grid)):
                    self.replan_agent(agent)

                next_pos = agent.get_next_pos()

                if next_pos != agent.current_pos:
                    nx_c, ny_c = int(next_pos[0]), int(next_pos[1])
                    cx_c, cy_c = int(agent.current_pos[0]), int(agent.current_pos[1])

                    # Immediate flame avoidance
                    if self.fire_grid[ny_c, nx_c]:
                        agent.entered_avoidance = True
                        # Local best step away from flames
                        best_pos, best_dist = None, float('inf')
                        for dx in [-2, 0, 2]:
                            for dy in [-2, 0, 2]:
                                if dx == 0 and dy == 0:
                                    continue
                                tx, ty = cx_c + dx, cy_c + dy
                                if 0 <= ty < self.h and 0 <= tx < self.w:
                                    if self.free_space_grid[ty, tx] and not self.fire_grid[ty, tx] and not self.occupancy_grid[ty, tx]:
                                        d = self.dist_map.get((float(tx), float(ty)), float('inf'))
                                        if d < best_dist:
                                            best_dist, best_pos = d, (tx, ty)
                        if best_pos:
                            self.occupancy_grid[cy_c, cx_c] = False
                            agent.current_pos = (float(best_pos[0]), float(best_pos[1]))
                            self.occupancy_grid[best_pos[1], best_pos[0]] = True
                            if best_dist < 15.0:
                                agent.is_evacuated = True
                                self.evacuated_count += 1
                                self.occupancy_grid[best_pos[1], best_pos[0]] = False

                    # Normal unblocked navigation
                    elif (0 <= ny_c < self.h and 0 <= nx_c < self.w and
                          self.free_space_grid[ny_c, nx_c] and
                          not self.occupancy_grid[ny_c, nx_c]):

                        self.occupancy_grid[cy_c, cx_c] = False
                        agent.move()

                        if not agent.is_evacuated:
                            new_cx, new_cy = int(agent.current_pos[0]), int(agent.current_pos[1])
                            self.occupancy_grid[new_cy, new_cx] = True
                        else:
                            self.evacuated_count += 1

            if not agent.is_evacuated:
                current_frame_positions.append((agent.current_pos[0], agent.current_pos[1], agent.agent_type))

        fire_pixels = np.argwhere(self.fire_grid)
        self.frame_history.append((current_frame_positions, fire_pixels))

    def is_complete(self) -> bool:
        return self.evacuated_count >= len(self.agents) or self.current_tick > self.sim_cfg.max_ticks

    def avoidance_fraction(self) -> float:
        if not self.agents:
            return 0.0
        return sum(1 for a in self.agents if a.entered_avoidance) / len(self.agents)


def run_proactive_simulation(nav_graph: nx.Graph, free_space_grid: np.ndarray, exit_nodes: list,
                             num_agents: int = 150, age_ratio: Tuple[int, int, int] = (6, 3, 1),
                             fire_start: Tuple[int, int] = (250, 580),
                             lookahead: int = 25, sensing_radius: float = 40.0) -> Tuple[List[int], int, float, int]:
    """
    Executes a single proactive simulation run and returns:
      (evac_history, total_ticks, avoidance_fraction, evacuated_count)
    """
    sim_cfg = SimulationConfig(num_agents=num_agents, age_ratio=age_ratio)
    rout_cfg = ProactiveRoutingConfig(lookahead_waypoints=lookahead, sensing_radius=sensing_radius)

    engine = ProactiveCrowdSimulationEngine(
        nav_graph, free_space_grid, exit_nodes,
        sim_config=sim_cfg, routing_config=rout_cfg,
        fire_mode='dynamic', fire_start=fire_start
    )

    # Populate agents from exit paths
    active_paths = engine.paths
    valid_spawn = [node for node in active_paths.keys() if nav_graph.nodes[node].get('type') == 'free']
    spawn_points = random.sample(valid_spawn, min(num_agents, len(valid_spawn)))

    a_pct, y_pct, e_pct = age_ratio[0] / sum(age_ratio), age_ratio[1] / sum(age_ratio), age_ratio[2] / sum(age_ratio)
    types = ['adult'] * int(num_agents * a_pct) + ['young'] * int(num_agents * y_pct) + ['elderly'] * int(num_agents * e_pct)
    while len(types) < len(spawn_points):
        types.append('adult')
    random.shuffle(types)

    for i, s_node in enumerate(spawn_points):
        path_nodes = active_paths[s_node][::-1]
        coords = [nav_graph.nodes[n]['pos'] for n in path_nodes]
        agent = ProactiveEvacuationAgent(
            agent_id=i, start_pos=coords[0], path=coords, agent_type=types[i],
            lookahead=lookahead, sensing_radius=sensing_radius
        )
        engine.add_agent(agent)

    history = []
    while not engine.is_complete():
        engine.step()
        history.append(engine.evacuated_count)

    return history, engine.current_tick, engine.avoidance_fraction(), engine.evacuated_count
