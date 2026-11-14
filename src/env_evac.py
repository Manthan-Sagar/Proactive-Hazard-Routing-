"""
env_evac.py
===========
Track B: Multi-Agent Gymnasium Environment for Hazard-Adaptive Indoor Evacuation.

BUG FIXES (Nov 2026 rebuild):
  1. Spawn fix: agents now spawn ONLY on graph nodes (dist_map keys), which are
     the navigable skeleton pixels connected to exits. The old code spawned on
     any free_space pixel (~90% of the floor plan image), causing "dots everywhere".
  2. Exit detection fix: replaced broken dist_map key lookup (which returned
     max_dist for off-grid positions) with direct Euclidean distance to exit coords.
  3. Added death_positions dict so video renderer can draw persistent death markers.
  4. Added curriculum_max_dist parameter for gradual spawn radius expansion.
  5. Increased default exit_reward from 100 to 300 for stronger learning signal.
"""

import math
import random
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

        # Demographic speed and social priority
        if agent_type == 'young':
            self.period = 3           # moves every 3 ticks (slower than adult)
            self.priority = 1         # yields first (gets out of the way)
            self.priority_norm = 0.33
            self.type_one_hot = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        elif agent_type == 'elderly':
            self.period = 4           # moves every 4 ticks (slowest)
            self.priority = 2         # yields second
            self.priority_norm = 0.66
            self.type_one_hot = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:  # adult
            self.period = 2           # moves every 2 ticks (fastest)
            self.priority = 3         # moves last (lets others yield first)
            self.priority_norm = 1.00
            self.type_one_hot = np.array([1.0, 0.0, 0.0], dtype=np.float32)


class HazardEvacEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # 9-DoF action space: Stay + 8 compass directions (step size 2px to match graph)
    ACTION_DELTAS = [
        (0,  0),   # 0: Stay
        (0, -2),   # 1: North
        (2, -2),   # 2: North-East
        (2,  0),   # 3: East
        (2,  2),   # 4: South-East
        (0,  2),   # 5: South
        (-2, 2),   # 6: South-West
        (-2, 0),   # 7: West
        (-2,-2),   # 8: North-West
    ]

    def __init__(
        self,
        free_space_grid: np.ndarray,
        dist_map: Dict[Tuple[float, float], float],
        exit_coords: List[Tuple[int, int]],
        num_agents: int = 50,
        age_ratio: Tuple[int, int, int] = (6, 3, 1),
        fire_spread_chance: float = 0.08,
        fire_interval: int = 5,
        obs_window_size: int = 21,
        max_ticks: int = 1500,
        fire_origin: Tuple[int, int] = (250, 580),
        curriculum_max_dist: Optional[float] = None,
        exit_reach_dist: float = 25.0,
        exit_reward: float = 300.0,
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
        self.curriculum_max_dist = curriculum_max_dist
        self.exit_reach_dist = exit_reach_dist
        self.exit_reward = exit_reward

        # Global spatial grids
        self.fire_grid        = np.zeros_like(self.free_space, dtype=bool)
        self.occupancy_grid   = np.zeros_like(self.free_space, dtype=bool)
        self.fire_dist_grid   = np.ones_like(self.free_space, dtype=np.float32)
        self.norm_potential_grid = np.zeros_like(self.free_space, dtype=np.float32)

        # Precompute normalized distance-to-exit potential field
        for (x, y), d in dist_map.items():
            ix, iy = int(round(x)), int(round(y))
            if 0 <= iy < self.h and 0 <= ix < self.w:
                self.norm_potential_grid[iy, ix] = float(d) / self.max_dist

        self.active_fire_front: set = set()
        self.current_tick: int = 0
        self.agents: Dict[int, EvacAgentState] = {}

        # --- SPAWN FIX: Use graph node positions (dist_map keys) instead of raw free_space ---
        # dist_map keys are ONLY positions on the navigable corridor skeleton that are
        # connected to at least one exit via the navigation graph. This is correct.
        self.graph_positions: List[Tuple[int, int]] = [
            (int(round(x)), int(round(y)))
            for (x, y) in dist_map.keys()
            if 0 <= int(round(y)) < self.h and 0 <= int(round(x)) < self.w
        ]

        # Death positions: persists until reset, used by video renderer to draw X markers
        self.death_positions: Dict[int, Tuple[int, int]] = {}

        # Gymnasium spaces
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Dict({
            "spatial":     spaces.Box(0.0, 1.0,   (obs_window_size, obs_window_size, 6), np.float32),
            "proprio":     spaces.Box(-1.0, 1.0,  (7,),                                  np.float32),
            "action_mask": spaces.Box(0, 1,        (9,),                                  np.int8),
        })

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

        # Ignite fire at designated chokepoint origin
        fx, fy = self.fire_origin
        if 0 <= fy < self.h and 0 <= fx < self.w:
            self.fire_grid[fy, fx] = True
            self.active_fire_front.add((fx, fy))
        self._update_fire_distances()

        # Determine curriculum spawn radius for this episode
        cur_dist = None
        if options and 'curriculum_max_dist' in options:
            cur_dist = options['curriculum_max_dist']
        elif self.curriculum_max_dist is not None:
            cur_dist = self.curriculum_max_dist

        # Build spawn pool from graph positions only
        if cur_dist is not None and self.exit_coords:
            spawn_pool = [
                (x, y) for (x, y) in self.graph_positions
                if not self.fire_grid[y, x]
                and min(math.hypot(x - ex, y - ey) for ex, ey in self.exit_coords) <= cur_dist
                and min(math.hypot(x - ex, y - ey) for ex, ey in self.exit_coords) > self.exit_reach_dist
            ]
            if len(spawn_pool) < max(self.num_agents, 10):
                spawn_pool = [(x, y) for (x, y) in self.graph_positions if not self.fire_grid[y, x]]
        else:
            spawn_pool = [(x, y) for (x, y) in self.graph_positions if not self.fire_grid[y, x]]

        if not spawn_pool:
            # Absolute last resort (should not happen with valid floor plan)
            vy, vx = np.where(self.free_space & ~self.fire_grid)
            spawn_pool = list(zip(vx.tolist(), vy.tolist()))

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

        # Spawn agents
        positions = random.sample(spawn_pool, min(self.num_agents, len(spawn_pool)))
        self.agents.clear()
        for i, (sx, sy) in enumerate(positions):
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
    # Fire dynamics
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
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
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
    # Observation helpers
    # ------------------------------------------------------------------
    def _get_exit_dist(self, x: int, y: int) -> float:
        """Graph distance to nearest exit with Euclidean fallback."""
        d = self.dist_map.get((float(x), float(y)), None)
        if d is not None:
            return d
        # Fallback: Euclidean to nearest exit coord
        if self.exit_coords:
            return min(math.hypot(x - ex, y - ey) for ex, ey in self.exit_coords)
        return self.max_dist

    def _get_action_mask(self, agent: EvacAgentState) -> np.ndarray:
        mask = np.zeros(9, dtype=np.int8)
        mask[0] = 1  # Stay is always valid
        ax, ay = agent.pos
        for a_idx, (dx, dy) in enumerate(self.ACTION_DELTAS):
            if a_idx == 0:
                continue
            tx, ty = ax + dx, ay + dy
            if 0 <= ty < self.h and 0 <= tx < self.w:
                if self.free_space[ty, tx] and not self.occupancy_grid[ty, tx]:
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
        spatial[dst_y1:dst_y2, dst_x1:dst_x2, 3] = self.fire_grid[src_y1:src_y2, src_x1:src_x2].astype(np.float32)
        spatial[dst_y1:dst_y2, dst_x1:dst_x2, 4] = self.fire_dist_grid[src_y1:src_y2, src_x1:src_x2]
        spatial[dst_y1:dst_y2, dst_x1:dst_x2, 5] = self.norm_potential_grid[src_y1:src_y2, src_x1:src_x2]

        # Proprioceptive vector: [adult?, young?, elderly?, speed_phase, exit_dist, fire_dist, dir_to_exit_x]
        exit_dist_norm = self._get_exit_dist(ax, ay) / self.max_dist
        fire_dist_norm = float(self.fire_dist_grid[ay, ax])
        speed_phase    = float(agent.ticks_remaining) / float(agent.period)

        # Direction unit vector toward nearest exit
        if self.exit_coords:
            nearest_exit = min(self.exit_coords, key=lambda e: math.hypot(ax - e[0], ay - e[1]))
            ex, ey = nearest_exit
        else:
            ex, ey = ax, ay
        vec_x, vec_y = ex - ax, ey - ay
        norm_v = math.hypot(vec_x, vec_y) + 1e-5
        ux = vec_x / norm_v

        proprio = np.array([
            agent.type_one_hot[0],  # is adult
            agent.type_one_hot[1],  # is young
            agent.type_one_hot[2],  # is elderly
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
    # Step
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

        # Process agents in social priority order: young first (they yield), then elderly, then adults
        active_ids = [aid for aid, a in self.agents.items() if not a.is_evacuated and not a.is_dead]
        active_ids.sort(key=lambda aid: (self.agents[aid].priority, random.random()))

        for aid in active_ids:
            agent  = self.agents[aid]
            agent.ticks_remaining += 1
            reward = -0.05  # small time penalty every tick

            if agent.ticks_remaining >= agent.period:
                agent.ticks_remaining = 0
                agent.total_steps += 1

                action    = actions.get(aid, 0)
                dx, dy    = self.ACTION_DELTAS[action]
                cx, cy    = agent.pos
                tx, ty    = cx + dx, cy + dy

                in_bounds = 0 <= ty < self.h and 0 <= tx < self.w

                # --- CASE 1: Stepped into fire → death ---
                if in_bounds and self.fire_grid[ty, tx]:
                    agent.is_dead = True
                    self.death_positions[aid] = (cx, cy)   # record death location for X marker
                    self.occupancy_grid[cy, cx] = False
                    reward -= 200.0
                    terminated[aid] = True

                # --- CASE 2: Wall / occupied collision → bump penalty ---
                elif not in_bounds or not self.free_space[ty, tx] or self.occupancy_grid[ty, tx]:
                    reward -= 1.5

                # --- CASE 3: Valid move ---
                else:
                    prev_pot = self._get_exit_dist(cx, cy)
                    curr_pot = self._get_exit_dist(tx, ty)

                    # Shaped progress reward: positive for moving toward exit
                    progress = max(0.0, prev_pot - curr_pot)
                    reward  += 2.0 * progress

                    # Hazard repulsion: exponential penalty for proximity to fire
                    fire_dist_px = self.fire_dist_grid[ty, tx] * 100.0
                    if fire_dist_px < 30.0:
                        reward -= 10.0 * math.exp(-fire_dist_px / 10.0)

                    # Execute the move
                    self.occupancy_grid[cy, cx] = False
                    agent.pos = (tx, ty)
                    self.occupancy_grid[ty, tx] = True

                    # --- EXIT DETECTION FIX: Euclidean proximity to any exit coord ---
                    # Old code used dist_map key lookup which silently failed for
                    # off-grid positions. Euclidean check is reliable.
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
