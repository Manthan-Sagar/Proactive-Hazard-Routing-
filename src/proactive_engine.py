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
import time
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

        # Demographic speeds & priorities matching Paper 2
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
                 fire_mode: str = 'dynamic', fire_start: Tuple[int, int] = (250, 580)):

        self.sim_cfg = sim_config or SimulationConfig()
        self.rout_cfg = routing_config or ProactiveRoutingConfig()

        self.nav_graph = nav_graph
        self.free_space_grid = free_space_grid
        self.exit_nodes = exit_nodes
        self.h, self.w = free_space_grid.shape

        self.occupancy_grid = np.zeros_like(free_space_grid, dtype=bool)
        self.fire_grid = np.zeros_like(free_space_grid, dtype=bool)
        self.hazard_buffer_grid = np.zeros_like(free_space_grid, dtype=bool)
        self.active_fire_front: Set[Tuple[int, int]] = set()

        self.fire_mode = fire_mode
        self.current_tick = 0
        self.evacuated_count = 0
        self.agents: List[ProactiveEvacuationAgent] = []
        self.frame_history = []
        self.last_dijkstra_tick = -100

        # Dynamic graph node tracking
        self.burned_graph_nodes: Set[str] = set()
        self.dist_map: Dict[Tuple[float, float], float] = {}
        self.paths: Dict[str, List[str]] = {}
        self.last_burned_count = 0

        # Initial graph escape paths
        self._compute_global_escape_paths()
        self.last_burned_count = len(self.burned_graph_nodes)

        if fire_mode != 'none':
            self.start_fire(int(fire_start[0]), int(fire_start[1]))

    def _compute_global_escape_paths(self):
        """Computes shortest paths on the unburned subgraph to the active exits."""
        valid_nodes = [n for n in self.nav_graph.nodes if n not in self.burned_graph_nodes]
        sub_g = self.nav_graph.subgraph(valid_nodes)
        active_exits = [e for e in self.exit_nodes if e in sub_g]
        if not active_exits:
            active_exits = self.exit_nodes

        try:
            lengths, paths = nx.multi_source_dijkstra(sub_g, sources=active_exits, weight='weight')
            self.paths = paths
            self.dist_map = {self.nav_graph.nodes[n]['pos']: d for n, d in lengths.items()}
            self.last_dijkstra_tick = self.current_tick
        except Exception as e:
            pass

    def start_fire(self, x: int, y: int):
        if 0 <= y < self.h and 0 <= x < self.w:
            self.fire_grid[y, x] = True
            self.active_fire_front.add((x, y))
            self._update_burned_graph_nodes([(x, y)])
            self._update_hazard_buffer()

    def _update_burned_graph_nodes(self, new_pixels):
        for px, py in new_pixels:
            nid = f"Ground_pix_{px}_{py}"
            if nid in self.nav_graph:
                self.burned_graph_nodes.add(nid)

    def _update_hazard_buffer(self):
        """Calculates repulsion buffer surrounding flames using Euclidean Distance Transform."""
        if not self.active_fire_front:
            self.hazard_buffer_grid[:] = False
            return
        dist = distance_transform_edt(~self.fire_grid)
        self.hazard_buffer_grid = (dist <= self.rout_cfg.hazard_buffer_margin) & (self.free_space_grid > 0)

    def spread_fire(self):
        new_front = set()
        new_pixels = []
        spread_chance = self.sim_cfg.fire_spread_chance
        for fx, fy in list(self.active_fire_front):
            has_unburned_neighbor = False
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx_c, ny_c = fx + dx, fy + dy
                if 0 <= ny_c < self.h and 0 <= nx_c < self.w:
                    if self.free_space_grid[ny_c, nx_c] and not self.fire_grid[ny_c, nx_c]:
                        has_unburned_neighbor = True
                        if random.random() < spread_chance:
                            self.fire_grid[ny_c, nx_c] = True
                            new_front.add((nx_c, ny_c))
                            new_pixels.append((nx_c, ny_c))
            if not has_unburned_neighbor:
                self.active_fire_front.remove((fx, fy))

        self.active_fire_front.update(new_front)
        if new_pixels:
            self._update_burned_graph_nodes(new_pixels)
            self._update_hazard_buffer()

    def add_agent(self, agent: ProactiveEvacuationAgent):
        self.agents.append(agent)
        cx, cy = int(agent.current_pos[0]), int(agent.current_pos[1])
        if 0 <= cy < self.h and 0 <= cx < self.w:
            self.occupancy_grid[cy, cx] = True

    def replan_agent(self, agent: ProactiveEvacuationAgent) -> bool:
        """
        Dynamically assigns a safe escape trajectory avoiding burned and buffer zones.
        """
        cx, cy = int(agent.current_pos[0]), int(agent.current_pos[1])
        best_node, best_d = None, float('inf')
        for dx in range(-4, 5, 2):
            for dy in range(-4, 5, 2):
                tx, ty = cx + dx, cy + dy
                nid = f"Ground_pix_{tx}_{ty}"
                if nid in self.paths:
                    d = np.hypot(dx, dy)
                    if d < best_d:
                        best_d, best_node = d, nid

        if best_node and best_node in self.paths:
            path_nodes = self.paths[best_node][::-1]
            coords = [self.nav_graph.nodes[n]['pos'] for n in path_nodes]
            if len(coords) > 1:
                agent.path = coords
                agent.path_index = 0
                agent.proactive_reroute_count += 1
                agent.entered_avoidance = True
                agent.last_reroute_tick = self.current_tick
                return True

        # If replanning failed, apply backoff cooldown to prevent hot-looping
        agent.last_reroute_tick = self.current_tick + 50
        return False

    def step(self):
        self.current_tick += 1

        # 1. Fire propagation
        if self.fire_mode == 'dynamic' and self.current_tick % self.sim_cfg.fire_interval == 0:
            self.spread_fire()

        active_agents = [a for a in self.agents if not a.is_evacuated]
        # Demographic priority sorting
        active_agents.sort(key=lambda a: (a.priority, random.random()))

        # 2. PROACTIVE VERIFICATION: Inspect upcoming path horizon
        needs_global_refresh = False
        compromised_agents = []
        fire_expanded = len(self.burned_graph_nodes) > self.last_burned_count

        for agent in active_agents:
            if self.current_tick - agent.last_reroute_tick >= self.rout_cfg.min_reroute_interval:
                if agent.check_downstream_hazard(self.fire_grid, self.hazard_buffer_grid):
                    compromised_agents.append(agent)
                    if fire_expanded:
                        needs_global_refresh = True

        # Debounced global distance field update (at most once every 50 ticks)
        if needs_global_refresh and (self.current_tick - self.last_dijkstra_tick >= 50):
            self._compute_global_escape_paths()
            self.last_burned_count = len(self.burned_graph_nodes)

        for agent in compromised_agents:
            self.replan_agent(agent)

        current_frame_positions = []

        # 3. Agent motion execution
        for agent in active_agents:
            agent.tick_counter += 1
            if agent.tick_counter >= agent.ticks_per_step:
                agent.tick_counter = 0

                next_pos = agent.get_next_pos()

                if next_pos != agent.current_pos:
                    nx_c, ny_c = int(next_pos[0]), int(next_pos[1])
                    cx_c, cy_c = int(agent.current_pos[0]), int(agent.current_pos[1])

                    # Immediate flame avoidance
                    if self.fire_grid[ny_c, nx_c]:
                        agent.entered_avoidance = True
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
