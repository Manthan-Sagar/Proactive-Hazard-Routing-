"""
env_evac.py
===========
Track B: Multi-Agent Gymnasium Environment for Hazard-Adaptive Indoor Evacuation.
Implements the formal Dec-POMDP formulation specified in docs/03_marl_architecture.md.
Features:
  - Multi-channel egocentric tensor observation (21 x 21 x 6)
  - Proprioceptive vector (R^7)
  - 9-DoF discrete spatial action space with action masking
  - Stochastic fire front propagation
  - Multi-objective reward shaping (goal, progress, hazard penalty, social yielding)
  - Heterogeneous demographic speed and priority gating
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

        if agent_type == 'young':
            self.period = 3
            self.priority = 1
            self.priority_norm = 0.33
            self.type_one_hot = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        elif agent_type == 'elderly':
            self.period = 4
            self.priority = 2
            self.priority_norm = 0.66
            self.type_one_hot = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else: # adult
            self.period = 2
            self.priority = 3
            self.priority_norm = 1.00
            self.type_one_hot = np.array([1.0, 0.0, 0.0], dtype=np.float32)


class HazardEvacEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Action offsets for 9-DoF navigation (step size 2 pixels)
    ACTION_DELTAS = [
        (0, 0),    # 0: Stay
        (0, -2),   # 1: North
        (2, -2),   # 2: North-East
        (2, 0),    # 3: East
        (2, 2),    # 4: South-East
        (0, 2),    # 5: South
        (-2, 2),   # 6: South-West
        (-2, 0),   # 7: West
        (-2, -2),  # 8: North-West
    ]

    def __init__(self, free_space_grid: np.ndarray, dist_map: Dict[Tuple[float, float], float],
                 exit_coords: List[Tuple[int, int]], num_agents: int = 50,
                 age_ratio: Tuple[int, int, int] = (6, 3, 1), fire_spread_chance: float = 0.08,
                 fire_interval: int = 5, obs_window_size: int = 21,
                 max_ticks: int = 1500, fire_origin: Tuple[int, int] = (250, 580)):

        super().__init__()

        self.free_space = free_space_grid.astype(bool)
        self.h, self.w = self.free_space.shape
        self.dist_map = dist_map
        self.max_dist = max(dist_map.values()) if dist_map else 1000.0
        self.exit_coords = exit_coords

        self.num_agents = num_agents
        self.age_ratio = age_ratio
        self.fire_spread_chance = fire_spread_chance
        self.fire_interval = fire_interval
        self.obs_window_size = obs_window_size
        self.half_window = obs_window_size // 2
        self.max_ticks = max_ticks
        self.fire_origin = fire_origin

        # Global spatial layers
        self.fire_grid = np.zeros_like(self.free_space, dtype=bool)
        self.occupancy_grid = np.zeros_like(self.free_space, dtype=bool)
        self.fire_dist_grid = np.zeros_like(self.free_space, dtype=np.float32)
        self.norm_potential_grid = np.zeros_like(self.free_space, dtype=np.float32)

        # Precompute normalized potential grid
        for (x, y), d in dist_map.items():
            ix, iy = int(x), int(y)
            if 0 <= iy < self.h and 0 <= ix < self.w:
                self.norm_potential_grid[iy, ix] = float(d) / self.max_dist

        self.active_fire_front = set()
        self.current_tick = 0
        self.agents: Dict[int, EvacAgentState] = {}

        # Gymnasium Spaces
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Dict({
            "spatial": spaces.Box(low=0.0, high=1.0, shape=(obs_window_size, obs_window_size, 6), dtype=np.float32),
            "proprio": spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(9,), dtype=np.int8)
        })

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[int, dict], Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.current_tick = 0
        self.fire_grid[:] = False
        self.occupancy_grid[:] = False
        self.active_fire_front.clear()

        # Start fire at designated origin
        fx, fy = self.fire_origin
        if 0 <= fy < self.h and 0 <= fx < self.w:
            self.fire_grid[fy, fx] = True
            self.active_fire_front.add((fx, fy))
        self._update_fire_distances()

        # Determine valid spawn locations
        valid_y, valid_x = np.where(self.free_space & ~self.fire_grid)
        valid_indices = list(range(len(valid_x)))
        spawn_indices = random.sample(valid_indices, min(self.num_agents, len(valid_indices)))

        # Assign demographics
        a_pct, y_pct, e_pct = self.age_ratio[0] / sum(self.age_ratio), self.age_ratio[1] / sum(self.age_ratio), self.age_ratio[2] / sum(self.age_ratio)
        types = ['adult'] * int(self.num_agents * a_pct) + ['young'] * int(self.num_agents * y_pct) + ['elderly'] * int(self.num_agents * e_pct)
        while len(types) < len(spawn_indices):
            types.append('adult')
        random.shuffle(types)

        self.agents.clear()
        for i, idx in enumerate(spawn_indices):
            sx, sy = int(valid_x[idx]), int(valid_y[idx])
            agent = EvacAgentState(i, (sx, sy), types[i])
            self.agents[i] = agent
            self.occupancy_grid[sy, sx] = True

        observations = {i: self._get_agent_observation(agent) for i, agent in self.agents.items()}
        info = {"evacuated": 0, "active": len(self.agents), "burned": 1}
        return observations, info

    def _update_fire_distances(self):
        if not self.active_fire_front:
            self.fire_dist_grid[:] = 1.0
            return
        dist = distance_transform_edt(~self.fire_grid)
        self.fire_dist_grid = np.clip(dist / 100.0, 0.0, 1.0).astype(np.float32)

    def _spread_fire(self):
        new_front = set()
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
                self.active_fire_front.remove((fx, fy))

        self.active_fire_front.update(new_front)
        if new_front:
            self._update_fire_distances()

    def _get_action_mask(self, agent: EvacAgentState) -> np.ndarray:
        mask = np.zeros(9, dtype=np.int8)
        mask[0] = 1 # Stay is always physically possible

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
        k = self.obs_window_size
        hw = self.half_window

        # Crop multi-channel window centered at (ax, ay)
        y_min, y_max = ay - hw, ay + hw + 1
        x_min, x_max = ax - hw, ax + hw + 1

        # Pad boundaries if window exceeds grid
        pad_top = max(0, -y_min)
        pad_bottom = max(0, y_max - self.h)
        pad_left = max(0, -x_min)
        pad_right = max(0, x_max - self.w)

        src_y1, src_y2 = max(0, y_min), min(self.h, y_max)
        src_x1, src_x2 = max(0, x_min), min(self.w, x_max)

        spatial_tensor = np.zeros((k, k, 6), dtype=np.float32)

        # Slices in target window
        dst_y1 = pad_top
        dst_y2 = k - pad_bottom
        dst_x1 = pad_left
        dst_x2 = k - pad_right

        # Channel 0: Traversability (Free Space)
        spatial_tensor[dst_y1:dst_y2, dst_x1:dst_x2, 0] = self.free_space[src_y1:src_y2, src_x1:src_x2].astype(np.float32)
        # Channel 1: Occupancy (Peer Presence)
        spatial_tensor[dst_y1:dst_y2, dst_x1:dst_x2, 1] = self.occupancy_grid[src_y1:src_y2, src_x1:src_x2].astype(np.float32)
        # Channel 3: Active Fire Front
        spatial_tensor[dst_y1:dst_y2, dst_x1:dst_x2, 3] = self.fire_grid[src_y1:src_y2, src_x1:src_x2].astype(np.float32)
        # Channel 4: Hazard Distance Transform
        spatial_tensor[dst_y1:dst_y2, dst_x1:dst_x2, 4] = self.fire_dist_grid[src_y1:src_y2, src_x1:src_x2]
        # Channel 5: Normalized Static Potential
        spatial_tensor[dst_y1:dst_y2, dst_x1:dst_x2, 5] = self.norm_potential_grid[src_y1:src_y2, src_x1:src_x2]

        # Proprioceptive Vector (R^7)
        # One-hot demographic (3), speed fraction (1), exit dist (1), fire dist (1), unit vector to exit (1)
        curr_dist_exit = self.dist_map.get((float(ax), float(ay)), self.max_dist) / self.max_dist
        curr_dist_fire = float(self.fire_dist_grid[ay, ax])
        ticks_frac = float(agent.ticks_remaining) / float(agent.period)

        # Nearest exit direction vector
        ex, ey = self.exit_coords[0] if self.exit_coords else (ax, ay)
        vec_x, vec_y = ex - ax, ey - ay
        norm = math.hypot(vec_x, vec_y) + 1e-5
        ux, uy = vec_x / norm, vec_y / norm

        proprio = np.array([
            agent.type_one_hot[0],
            agent.type_one_hot[1],
            agent.type_one_hot[2],
            ticks_frac,
            curr_dist_exit,
            curr_dist_fire,
            ux
        ], dtype=np.float32)

        mask = self._get_action_mask(agent)

        return {
            "spatial": spatial_tensor,
            "proprio": proprio,
            "action_mask": mask
        }

    def step(self, actions: Dict[int, int]) -> Tuple[Dict[int, dict], Dict[int, float], Dict[int, bool], Dict[int, bool], Dict[str, Any]]:
        self.current_tick += 1

        if self.current_tick % self.fire_interval == 0:
            self._spread_fire()

        rewards: Dict[int, float] = {}
        terminated: Dict[int, bool] = {}
        truncated: Dict[int, bool] = {}
        observations: Dict[int, dict] = {}

        # Sort agents by demographic yielding priority
        active_ids = [aid for aid, a in self.agents.items() if not a.is_evacuated and not a.is_dead]
        active_ids.sort(key=lambda aid: (self.agents[aid].priority, random.random()))

        for aid in active_ids:
            agent = self.agents[aid]
            agent.ticks_remaining += 1
            reward = -0.05 # Step time cost

            # Demographic speed gating: agent only moves when ticks_remaining >= period
            if agent.ticks_remaining >= agent.period:
                agent.ticks_remaining = 0
                agent.total_steps += 1
                action = actions.get(aid, 0)
                dx, dy = self.ACTION_DELTAS[action]
                cx, cy = agent.pos
                tx, ty = cx + dx, cy + dy

                # 1. Fatal Fire Collision Check
                if 0 <= ty < self.h and 0 <= tx < self.w and self.fire_grid[ty, tx]:
                    agent.is_dead = True
                    self.occupancy_grid[cy, cx] = False
                    reward -= 200.0
                    terminated[aid] = True

                # 2. Obstacle / Occupied Collision Check
                elif not (0 <= ty < self.h and 0 <= tx < self.w) or not self.free_space[ty, tx] or self.occupancy_grid[ty, tx]:
                    reward -= 1.5 # Bump penalty

                # 3. Successful Valid Step
                else:
                    prev_pot = self.dist_map.get((float(cx), float(cy)), self.max_dist)
                    curr_pot = self.dist_map.get((float(tx), float(ty)), self.max_dist)

                    # Progress reward
                    progress = max(0.0, prev_pot - curr_pot)
                    reward += 2.0 * progress

                    # Hazard repulsion penalty: -kappa * exp(-d / lambda)
                    fire_dist_px = self.fire_dist_grid[ty, tx] * 100.0
                    if fire_dist_px < 30.0:
                        reward -= 10.0 * math.exp(-fire_dist_px / 10.0)

                    # Execute move
                    self.occupancy_grid[cy, cx] = False
                    agent.pos = (tx, ty)
                    self.occupancy_grid[ty, tx] = True

                    # 4. Exit Reached Check
                    if curr_pot < 15.0:
                        agent.is_evacuated = True
                        self.occupancy_grid[ty, tx] = False
                        reward += 100.0
                        terminated[aid] = True

            rewards[aid] = reward
            terminated[aid] = terminated.get(aid, agent.is_evacuated or agent.is_dead)
            truncated[aid] = self.current_tick >= self.max_ticks

            if not terminated[aid]:
                observations[aid] = self._get_agent_observation(agent)

        all_done = (
            all(a.is_evacuated or a.is_dead for a in self.agents.values())
            or self.current_tick >= self.max_ticks
        )

        info = {
            "tick": self.current_tick,
            "evacuated": sum(1 for a in self.agents.values() if a.is_evacuated),
            "dead": sum(1 for a in self.agents.values() if a.is_dead),
            "active": sum(1 for a in self.agents.values() if not a.is_evacuated and not a.is_dead),
            "all_done": all_done
        }

        return observations, rewards, terminated, truncated, info

