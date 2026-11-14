"""
env_evac.py
===========
Track B: Multi-Agent Gymnasium Environment for Hazard-Adaptive Indoor Evacuation.

HIGH-ENERGY / SPICY REBUILD (Nov 2026):
  1. Fast Movement: Step size increased to 4px (with continuous midpoint wall check).
     Adults move every tick (period=1), young every 2 ticks, elderly every 2 ticks.
     Agents can now easily traverse the full 701px building in 150-200 ticks!
  2. Aggressive Fire Spread: Fire expands faster across corridor chokepoints
     (spread_chance=0.20, interval=2), creating urgent life-or-death pressure.
  3. Stratified Random Spawning: Agents spawn randomly across 4 distinct vertical zones
     (North shops, mid passage, central corridor, south vestibule).
  4. Exit Detection: Euclidean reach radius of 30px with +500 reward.
  5. Persistent Death Tracking: Records exact demise coordinates for vivid death rendering.
"""

import math
import random
import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.ndimage import distance_transform_edt
from typing import Dict, List, Tuple, Any, Optional


class EvacAgentState:
    def __init__(self, agent_id: int, pos: Tuple[int, int], agent_type: str = 'adult'):
        self.agent_id = agent_id
        self.pos = pos
        self.agent_type = agent_type
        self.is_evacuated = False
        self.is_dead = False
        self.ticks_remaining = 0
        self.total_steps = 0
        self.last_action = 0
        self.wall_stick_ticks = 0  # Consecutive ticks staying near/against walls

        # Fast demographic speeds & yielding priorities
        if agent_type == 'young':
            self.period = 1           # agile youth (moves every tick)
            self.priority = 1         # yields first
            self.priority_norm = 0.33
            self.type_one_hot = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        elif agent_type == 'elderly':
            self.period = 2           # moves every 2 ticks (steady pace)
            self.priority = 2         # yields second
            self.priority_norm = 0.66
            self.type_one_hot = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:  # adult
            self.period = 1           # moves EVERY tick (fast!)
            self.priority = 3         # highest speed
            self.priority_norm = 1.00
            self.type_one_hot = np.array([1.0, 0.0, 0.0], dtype=np.float32)


class HazardEvacEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # 9-DoF action space: Stay + 8 compass directions with FAST 4-pixel step size
    ACTION_DELTAS = [
        (0,  0),   # 0: Stay
        (0, -4),   # 1: North
        (3, -3),   # 2: North-East
        (4,  0),   # 3: East
        (3,  3),   # 4: South-East
        (0,  4),   # 5: South
        (-3, 3),   # 6: South-West
        (-4, 0),   # 7: West
        (-3,-3),   # 8: North-West
    ]

    def __init__(
        self,
        free_space_grid: np.ndarray,
        dist_map: Dict[Tuple[float, float], float],
        exit_coords: List[Tuple[int, int]],
        num_agents: int = 15,
        age_ratio: Tuple[int, int, int] = (6, 3, 1),
        fire_spread_chance: float = 0.20,
        fire_interval: int = 2,
        obs_window_size: int = 21,
        max_ticks: int = 300,
        fire_origin: Tuple[int, int] = (250, 560),
        exit_reach_dist: float = 30.0,
        exit_reward: float = 500.0,
    ):
        super().__init__()

        self.free_space = free_space_grid.astype(bool)
        self.h, self.w = self.free_space.shape
        self.dist_map = dist_map
        self.max_dist = max(dist_map.values()) if dist_map else 1000.0
        self.exit_coords = list(exit_coords)

        self.num_agents = num_agents
        self.age_ratio = age_ratio
        self.fire_spread_chance = fire_spread_chance
        self.fire_interval = fire_interval
        self.obs_window_size = obs_window_size
        self.half_window = obs_window_size // 2
        self.max_ticks = max_ticks
        self.fire_origin = fire_origin
        self.exit_reach_dist = exit_reach_dist
        self.exit_reward = exit_reward

        # Global spatial grids
        self.fire_grid           = np.zeros_like(self.free_space, dtype=bool)
        self.occupancy_grid      = np.zeros_like(self.free_space, dtype=bool)
        self.fire_dist_grid      = np.ones_like(self.free_space, dtype=np.float32)
        # Precompute distance from walls (px) for all navigable space
        self.wall_dist_grid      = distance_transform_edt(self.free_space).astype(np.float32)

        # Build dense 2D Dijkstra potential grid over all free space
        # Smoothly interpolates across all pixels, eliminating odd-coordinate lookup misses
        self.dense_potential_grid = np.full((self.h, self.w), fill_value=self.max_dist, dtype=np.float32)
        hh, hw = (self.h + 1) // 2, (self.w + 1) // 2
        half_grid = np.zeros((hh, hw), dtype=np.float32)
        for (x, y), d in dist_map.items():
            ix, iy = int(round(x)) // 2, int(round(y)) // 2
            if 0 <= iy < hh and 0 <= ix < hw:
                half_grid[iy, ix] = float(d)

        full_interp = cv2.resize(half_grid, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        self.dense_potential_grid[self.free_space] = full_interp[self.free_space]

        # Normalized distance-to-exit potential field for CNN Channel 5 (smooth across all pixels)
        self.norm_potential_grid = np.zeros_like(self.free_space, dtype=np.float32)
        self.norm_potential_grid[self.free_space] = np.clip(
            self.dense_potential_grid[self.free_space] / self.max_dist, 0.0, 1.0
        )

        self.active_fire_front: set = set()
        self.current_tick: int = 0
        self.agents: Dict[int, EvacAgentState] = {}

        # Navigable corridor skeleton nodes connected to exits
        self.graph_positions: List[Tuple[int, int]] = [
            (int(round(x)), int(round(y)))
            for (x, y) in dist_map.keys()
            if 0 <= int(round(y)) < self.h and 0 <= int(round(x)) < self.w
        ]

        # Stratify graph positions into 4 vertical zones, requiring strong wall clearance (>= 6.0px)
        # so agents NEVER spawn adjacent to or overlapping walls
        self.zone_positions: Dict[int, List[Tuple[int, int]]] = {
            1: [(x, y) for (x, y) in self.graph_positions if 40 <= y < 200 and self.wall_dist_grid[y, x] >= 6.0],
            2: [(x, y) for (x, y) in self.graph_positions if 200 <= y < 360 and self.wall_dist_grid[y, x] >= 6.0],
            3: [(x, y) for (x, y) in self.graph_positions if 360 <= y < 520 and self.wall_dist_grid[y, x] >= 6.0],
            4: [(x, y) for (x, y) in self.graph_positions if 520 <= y < 620 and self.wall_dist_grid[y, x] >= 6.0],
        }

        # Death tracking: stores (x, y) where each agent died
        self.death_positions: Dict[int, Tuple[int, int]] = {}

        # Gymnasium spaces
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Dict({
            "spatial":     spaces.Box(0.0, 1.0,  (obs_window_size, obs_window_size, 6), np.float32),
            "proprio":     spaces.Box(-1.0, 1.0, (7,),                                  np.float32),
            "action_mask": spaces.Box(0, 1,       (9,),                                  np.int8),
        })

    # ------------------------------------------------------------------
    # Step Validation (Midpoint check prevents jumping through thin walls)
    # ------------------------------------------------------------------
    def is_valid_step(self, cx: int, cy: int, tx: int, ty: int) -> bool:
        if not (0 <= ty < self.h and 0 <= tx < self.w):
            return False
        # Staying in place is valid if current cell is free space
        if tx == cx and ty == cy:
            return self.free_space[ty, tx]
        if not self.free_space[ty, tx] or self.occupancy_grid[ty, tx]:
            return False
        # Physical body clearance: prevent stepping directly against/inside walls (< 2.0px)
        if self.wall_dist_grid[ty, tx] < 2.0:
            if not self.exit_coords or min(math.hypot(tx - ex, ty - ey) for ex, ey in self.exit_coords) > 50.0:
                return False
        # Midpoint wall collision check
        mx, my = (cx + tx) // 2, (cy + ty) // 2
        if not self.free_space[my, mx]:
            return False
        return True

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[Dict[int, dict], Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.current_tick = 0
        self.fire_grid[:] = False
        self.occupancy_grid[:] = False
        self.active_fire_front.clear()
        self.death_positions.clear()

        # Ignite fire at the corridor bottleneck chokepoint
        fx, fy = self.fire_origin
        if 0 <= fy < self.h and 0 <= fx < self.w:
            # Seed a 3x3 fire cluster for immediate threat
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if 0 <= fy + dy < self.h and 0 <= fx + dx < self.w and self.free_space[fy + dy, fx + dx]:
                        self.fire_grid[fy + dy, fx + dx] = True
                        self.active_fire_front.add((fx + dx, fy + dy))
        self._update_fire_distances()

        # Stratified random spawning across all 4 zones (keeping safe clearance from fire origin)
        chosen_spawns: List[Tuple[int, int]] = []
        per_zone = max(1, self.num_agents // 4)
        for z in [1, 2, 3, 4]:
            pool = [
                p for p in self.zone_positions[z]
                if math.hypot(p[0] - fx, p[1] - fy) >= 45.0 and not self.fire_grid[p[1], p[0]]
            ]
            if pool:
                take = min(per_zone, len(pool))
                chosen_spawns.extend(random.sample(pool, take))

        # Fill any remainder from the clear graph pool (>= 6.0px from wall, >= 45px from fire origin)
        clear_pool = [
            p for p in self.graph_positions
            if self.wall_dist_grid[p[1], p[0]] >= 6.0 and math.hypot(p[0] - fx, p[1] - fy) >= 45.0
        ]
        while len(chosen_spawns) < self.num_agents:
            cand = random.choice(clear_pool if clear_pool else self.graph_positions)
            if cand not in chosen_spawns and not self.fire_grid[cand[1], cand[0]]:
                chosen_spawns.append(cand)

        chosen_spawns = chosen_spawns[:self.num_agents]

        # Assign demographic types
        total_w = sum(self.age_ratio)
        n_adult   = int(self.num_agents * self.age_ratio[0] / total_w)
        n_young   = int(self.num_agents * self.age_ratio[1] / total_w)
        n_elderly = max(0, self.num_agents - n_adult - n_young)
        types = ['adult'] * n_adult + ['young'] * n_young + ['elderly'] * n_elderly
        while len(types) < self.num_agents:
            types.append('adult')
        types = types[:self.num_agents]
        random.shuffle(types)

        # Initialize agents
        self.agents.clear()
        for i, (sx, sy) in enumerate(chosen_spawns):
            agent = EvacAgentState(i, (sx, sy), types[i])
            self.agents[i] = agent
            self.occupancy_grid[sy, sx] = True

        observations = {i: self._get_agent_observation(a) for i, a in self.agents.items()}
        info = {
            "tick": 0,
            "evacuated": 0,
            "dead": 0,
            "active": len(self.agents),
            "all_done": False,
        }
        return observations, info

    # ------------------------------------------------------------------
    # Fire Dynamics
    # ------------------------------------------------------------------
    def _update_fire_distances(self):
        if not self.active_fire_front:
            self.fire_dist_grid[:] = 1.0
            return
        dist = distance_transform_edt(~self.fire_grid)
        self.fire_dist_grid = np.clip(dist / 100.0, 0.0, 1.0).astype(np.float32)

    def _spread_fire(self):
        new_front: set = set()
        for fx, fy in list(self.active_fire_front):
            has_unburned = False
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                nx_c, ny_c = fx + dx, fy + dy
                if 0 <= ny_c < self.h and 0 <= nx_c < self.w:
                    if self.free_space[ny_c, nx_c] and not self.fire_grid[ny_c, nx_c]:
                        has_unburned = True
                        if random.random() < self.fire_spread_chance:
                            self.fire_grid[ny_c, nx_c] = True
                            new_front.add((nx_c, ny_c))
            if not has_unburned:
                self.active_fire_front.discard((fx, fy))
        self.active_fire_front.update(new_front)
        if new_front:
            self._update_fire_distances()

    # ------------------------------------------------------------------
    # Observation & Action Masking
    # ------------------------------------------------------------------
    def _get_exit_dist(self, x: int, y: int) -> float:
        ix, iy = int(round(x)), int(round(y))
        if 0 <= iy < self.h and 0 <= ix < self.w and self.free_space[iy, ix]:
            d = self.dense_potential_grid[iy, ix]
            if d < self.max_dist and d >= 0:
                return float(d)
        d_dict = self.dist_map.get((float(x), float(y)), None)
        if d_dict is not None:
            return float(d_dict)
        if self.exit_coords:
            return min(math.hypot(x - ex, y - ey) for ex, ey in self.exit_coords)
        return self.max_dist

    def _get_action_mask(self, agent: EvacAgentState) -> np.ndarray:
        mask = np.zeros(9, dtype=np.int8)
        mask[0] = 1  # Stay is always possible
        ax, ay = agent.pos
        exit_dist = self._get_exit_dist(ax, ay)
        near_exit = exit_dist <= 60.0

        for a_idx, (dx, dy) in enumerate(self.ACTION_DELTAS):
            if a_idx == 0:
                continue
            tx, ty = ax + dx, ay + dy
            if not self.is_valid_step(ax, ay, tx, ty):
                continue

            # Hard wall clearance masking:
            # Forbid stepping into positions touching/hugging walls (< 3.0px) unless near an exit
            if near_exit or self.wall_dist_grid[ty, tx] >= 3.0:
                mask[a_idx] = 1

        # Fallback: if wall clearance masked out all non-stay moves (e.g. narrow doorways),
        # allow all valid free-space steps so agent is never trapped
        if mask[1:].sum() == 0:
            for a_idx, (dx, dy) in enumerate(self.ACTION_DELTAS):
                if a_idx == 0:
                    continue
                tx, ty = ax + dx, ay + dy
                if self.is_valid_step(ax, ay, tx, ty):
                    mask[a_idx] = 1

        return mask

    def _get_agent_observation(self, agent: EvacAgentState) -> Dict[str, np.ndarray]:
        ax, ay = agent.pos
        k  = self.obs_window_size
        hw = self.half_window

        y_min, y_max = ay - hw, ay + hw + 1
        x_min, x_max = ax - hw, ax + hw + 1

        pad_top    = max(0, -y_min)
        pad_bottom = max(0,  y_max - self.h)
        pad_left   = max(0, -x_min)
        pad_right  = max(0,  x_max - self.w)

        src_y1, src_y2 = max(0, y_min), min(self.h, y_max)
        src_x1, src_x2 = max(0, x_min), min(self.w, x_max)

        dst_y1 = pad_top;       dst_y2 = k - pad_bottom
        dst_x1 = pad_left;      dst_x2 = k - pad_right

        spatial = np.zeros((k, k, 6), dtype=np.float32)
        spatial[dst_y1:dst_y2, dst_x1:dst_x2, 0] = self.free_space[src_y1:src_y2, src_x1:src_x2].astype(np.float32)
        spatial[dst_y1:dst_y2, dst_x1:dst_x2, 1] = self.occupancy_grid[src_y1:src_y2, src_x1:src_x2].astype(np.float32)
        spatial[dst_y1:dst_y2, dst_x1:dst_x2, 2] = np.clip(self.wall_dist_grid[src_y1:src_y2, src_x1:src_x2] / 20.0, 0.0, 1.0)
        spatial[dst_y1:dst_y2, dst_x1:dst_x2, 3] = self.fire_grid[src_y1:src_y2, src_x1:src_x2].astype(np.float32)
        spatial[dst_y1:dst_y2, dst_x1:dst_x2, 4] = self.fire_dist_grid[src_y1:src_y2, src_x1:src_x2]
        spatial[dst_y1:dst_y2, dst_x1:dst_x2, 5] = self.norm_potential_grid[src_y1:src_y2, src_x1:src_x2]

        exit_dist_norm = self._get_exit_dist(ax, ay) / self.max_dist
        fire_dist_norm = float(self.fire_dist_grid[ay, ax])
        speed_phase    = float(agent.ticks_remaining) / float(agent.period)

        if self.exit_coords:
            nearest_exit = min(self.exit_coords, key=lambda e: math.hypot(ax - e[0], ay - e[1]))
            ex, ey = nearest_exit
        else:
            ex, ey = ax, ay
        vec_x, vec_y = ex - ax, ey - ay
        norm_v = math.hypot(vec_x, vec_y) + 1e-5
        ux = vec_x / norm_v

        proprio = np.array([
            agent.type_one_hot[0],
            agent.type_one_hot[1],
            agent.type_one_hot[2],
            speed_phase,
            exit_dist_norm,
            fire_dist_norm,
            ux,
        ], dtype=np.float32)

        return {
            "spatial":     spatial,
            "proprio":     proprio,
            "action_mask": self._get_action_mask(agent),
        }

    # ------------------------------------------------------------------
    # Step Execution
    # ------------------------------------------------------------------
    def step(
        self,
        actions: Dict[int, int]
    ) -> Tuple[Dict[int, dict], Dict[int, float], Dict[int, bool], Dict[int, bool], Dict[str, Any]]:
        self.current_tick += 1

        if self.current_tick % self.fire_interval == 0:
            self._spread_fire()

        rewards:     Dict[int, float] = {}
        terminated:  Dict[int, bool]  = {}
        truncated:   Dict[int, bool]  = {}
        observations: Dict[int, dict] = {}

        # Social priority sorting: young first, then elderly, then adults
        active_ids = [aid for aid, a in self.agents.items() if not a.is_evacuated and not a.is_dead]
        active_ids.sort(key=lambda aid: (self.agents[aid].priority, random.random()))

        for aid in active_ids:
            agent  = self.agents[aid]
            agent.ticks_remaining += 1
            reward = -0.02  # base time step cost

            if agent.ticks_remaining >= agent.period:
                agent.ticks_remaining = 0
                agent.total_steps += 1

                action    = actions.get(aid, 0)
                agent.last_action = action
                dx, dy    = self.ACTION_DELTAS[action]
                cx, cy    = agent.pos
                tx, ty    = cx + dx, cy + dy

                in_bounds = 0 <= ty < self.h and 0 <= tx < self.w

                # 1. FATAL FIRE COLLISION (Death)
                # Check target and midpoint for fire
                mx, my = (cx + tx) // 2, (cy + ty) // 2
                stepped_in_fire = in_bounds and (self.fire_grid[ty, tx] or self.fire_grid[my, mx])

                if stepped_in_fire:
                    agent.is_dead = True
                    self.death_positions[aid] = (cx, cy)
                    self.occupancy_grid[cy, cx] = False
                    reward -= 150.0  # Clear fire demise penalty
                    terminated[aid] = True

                # 2. INVALID / WALL STEP
                elif not self.is_valid_step(cx, cy, tx, ty):
                    reward -= 1.0

                # 3. VOLUNTARY IDLING (Anti-freeze penalty: don't stand still in burning building)
                elif action == 0:
                    reward -= 0.25

                # 4. SUCCESSFUL FORWARD STEP
                else:
                    prev_pot = self._get_exit_dist(cx, cy)
                    curr_pot = self._get_exit_dist(tx, ty)

                    # Progress reward: positive incentive to advance toward exit
                    progress = prev_pot - curr_pot
                    reward  += 3.0 * progress

                    # Hazard proximity penalty: exponential repulsion from fire
                    fire_dist_px = self.fire_dist_grid[ty, tx] * 100.0
                    if fire_dist_px < 35.0:
                        reward -= 12.0 * math.exp(-fire_dist_px / 12.0)

                    # Gentle wall proximity shaping: prefer corridor center
                    if self.wall_dist_grid[ty, tx] < 4.0:
                        reward -= 0.20

                    # Update position
                    self.occupancy_grid[cy, cx] = False
                    agent.pos = (tx, ty)
                    self.occupancy_grid[ty, tx] = True

                    # 5. EXIT REACHED CHECK
                    if self.exit_coords:
                        min_exit_dist = min(math.hypot(tx - ex, ty - ey) for ex, ey in self.exit_coords)
                        if min_exit_dist <= self.exit_reach_dist:
                            agent.is_evacuated = True
                            self.occupancy_grid[ty, tx] = False
                            reward += self.exit_reward
                            terminated[aid] = True

            rewards[aid]    = reward
            terminated[aid] = terminated.get(aid, agent.is_evacuated or agent.is_dead)
            truncated[aid]  = self.current_tick >= self.max_ticks

            if not terminated[aid]:
                observations[aid] = self._get_agent_observation(agent)

        all_done = (
            all(a.is_evacuated or a.is_dead for a in self.agents.values())
            or self.current_tick >= self.max_ticks
        )

        info = {
            "tick":      self.current_tick,
            "evacuated": sum(1 for a in self.agents.values() if a.is_evacuated),
            "dead":      sum(1 for a in self.agents.values() if a.is_dead),
            "active":    sum(1 for a in self.agents.values() if not a.is_evacuated and not a.is_dead),
            "all_done":  all_done,
        }

        return observations, rewards, terminated, truncated, info
