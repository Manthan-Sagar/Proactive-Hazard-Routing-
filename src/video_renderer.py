"""
video_renderer.py
=================
Renders simulation frames for the MARL evacuation environment.

REBUILD (Nov 2026):
  - Bigger agent dots (radius 7 + white border) — visible on large floor plans
  - High-contrast colors: Green=Adult, Cyan=Young, Orange=Elderly, Red X=Dead
  - Persistent death markers drawn at every frame after agent dies
  - Full agent trail (last 60 steps, demographic color, fading alpha)
  - Bottom legend bar on every frame explaining colors
  - Two-line banner showing tick, episode title, evacuated/dead/active counts
  - EXIT labels drawn near exit circles
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import networkx as nx

from src.floor_loader import load_floor
from src.env_evac import HazardEvacEnv
from src.rl_model import ActorCriticNetwork, ActorCriticRNN


# ---------------------------------------------------------------------------
# Color palette  (all BGR for OpenCV)
# ---------------------------------------------------------------------------
COLORS = {
    'adult':   (0,   240,  60),    # Bright lime green
    'young':   (255, 210,   0),    # Vivid cyan
    'elderly': (0,   140, 255),    # High-vis bright orange
    'dead':    (0,   0,   255),    # Vivid red
    'exit':    (0,   220,   0),    # Emerald green
    'fire':    (20,   30, 255),    # Flame orange-red
    'smoke':   np.array([80, 60, 20], dtype=np.float32),
}

BANNER_H  = 44   # top info bar height (px)
LEGEND_H  = 34   # bottom legend bar height (px)
DOT_R     = 4    # agent dot radius (compact 4px core + 2px halo stays strictly inside corridor walls)
TRAIL_LEN = 60   # max trail history shown


# ---------------------------------------------------------------------------
# Main frame drawing function
# ---------------------------------------------------------------------------
def draw_simulation_frame(
    base_img: np.ndarray,
    env: HazardEvacEnv,
    title: str = "",
    agent_trails: Optional[Dict[int, List[Tuple[int, int]]]] = None,
) -> np.ndarray:
    """
    Renders one full frame:
      banner | floor plan + fire + agents + trails + deaths | legend
    Returns a BGR numpy array ready for cv2.imwrite or VideoWriter.
    """
    # Ensure writeable BGR copy
    if base_img.ndim == 2:
        frame = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    elif base_img.shape[2] == 4:
        frame = cv2.cvtColor(base_img, cv2.COLOR_RGBA2BGR)
    else:
        frame = np.ascontiguousarray(base_img[:, :, ::-1] if base_img.dtype == np.uint8 else base_img.copy())
        # Ensure RGB→BGR if loaded with PIL
        frame = frame.copy()

    H, W = frame.shape[:2]

    # ------------------------------------------------------------------
    # 1. Smoke buffer overlay (semi-transparent dark blue near fire)
    # ------------------------------------------------------------------
    smoke_mask = (env.fire_dist_grid > 0.0) & (env.fire_dist_grid < 0.30) & ~env.fire_grid
    if smoke_mask.any():
        frame[smoke_mask] = np.clip(
            frame[smoke_mask].astype(np.float32) * 0.55 + COLORS['smoke'] * 0.45, 0, 255
        ).astype(np.uint8)

    # ------------------------------------------------------------------
    # 2. Active fire (bright fill)
    # ------------------------------------------------------------------
    frame[env.fire_grid] = COLORS['fire']

    # ------------------------------------------------------------------
    # 3. Exit markers (prominent green circle + white ring + "EXIT" label)
    # ------------------------------------------------------------------
    for idx, (ex, ey) in enumerate(env.exit_coords):
        px, py = int(ex), int(ey)
        cv2.circle(frame, (px, py), 13, (255, 255, 255), -1)
        cv2.circle(frame, (px, py), 11, COLORS['exit'], -1)
        cv2.putText(frame, f"EXIT {idx+1}", (px - 20, py - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # 4. Agent trails (demographic color, fade older steps)
    # ------------------------------------------------------------------
    if agent_trails:
        for aid, trail in agent_trails.items():
            agent = env.agents.get(aid)
            if agent is None or len(trail) < 2:
                continue
            base_color = COLORS.get(agent.agent_type, COLORS['adult'])
            tail = trail[-TRAIL_LEN:] if len(trail) > TRAIL_LEN else trail
            n = len(tail)
            for k in range(n - 1):
                alpha = 0.15 + 0.85 * (k + 1) / n   # 15%→100% brightness along trail
                seg_color = tuple(int(c * alpha * 0.6) for c in base_color)
                cv2.line(frame, tail[k], tail[k + 1], seg_color, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # 5. Persistent death markers (high-contrast shield + red skull cross)
    # Guaranteed to pop clearly on top of fire and smoke
    # ------------------------------------------------------------------
    for aid, (dx, dy) in env.death_positions.items():
        cx, cy = int(dx), int(dy)
        # 1. Solid white outer shield
        cv2.circle(frame, (cx, cy), 13, (255, 255, 255), -1)
        # 2. Pitch black inner circular badge
        cv2.circle(frame, (cx, cy), 10, (12, 12, 12), -1)
        # 3. Vivid red anti-aliased cross
        cv2.line(frame, (cx - 7, cy - 7), (cx + 7, cy + 7), COLORS['dead'], 3, cv2.LINE_AA)
        cv2.line(frame, (cx - 7, cy + 7), (cx + 7, cy - 7), COLORS['dead'], 3, cv2.LINE_AA)
        # 4. Center yellow core dot
        cv2.circle(frame, (cx, cy), 2, (0, 255, 255), -1)
        # 5. Text tag above marker
        cv2.putText(frame, "DEAD", (cx - 14, cy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 255), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # 6. Live agents (large dot + white border ring)
    # ------------------------------------------------------------------
    for aid, agent in env.agents.items():
        if agent.is_dead or agent.is_evacuated:
            continue
        cx, cy = int(agent.pos[0]), int(agent.pos[1])
        color = COLORS.get(agent.agent_type, COLORS['adult'])
        cv2.circle(frame, (cx, cy), DOT_R + 2, (255, 255, 255), -1)   # sharp 2px white halo
        cv2.circle(frame, (cx, cy), DOT_R,     color, -1)              # demographic fill

    # ------------------------------------------------------------------
    # 7. Top banner: 2 lines of info
    # ------------------------------------------------------------------
    banner = np.zeros((BANNER_H, W, 3), dtype=np.uint8)
    banner[:] = (20, 20, 20)

    n_evac    = sum(1 for a in env.agents.values() if a.is_evacuated)
    n_dead    = sum(1 for a in env.agents.values() if a.is_dead)
    n_active  = sum(1 for a in env.agents.values() if not a.is_evacuated and not a.is_dead)
    n_total   = len(env.agents)
    evac_pct  = int(100 * n_evac / n_total) if n_total > 0 else 0

    line1 = f"Tick {env.current_tick:03d}/{env.max_ticks}    {title}"
    line2 = (f"Evacuated: {n_evac}/{n_total} ({evac_pct}%)   "
             f"Dead: {n_dead}   Active: {n_active}")

    cv2.putText(banner, line1, (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(banner, line2, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # 8. Bottom legend bar
    # ------------------------------------------------------------------
    legend = np.full((LEGEND_H, W, 3), (28, 28, 28), dtype=np.uint8)
    legend_items = [
        ("Adult",    COLORS['adult']),
        ("Young",    COLORS['young']),
        ("Elderly",  COLORS['elderly']),
        ("Dead",     COLORS['dead']),
    ]
    x_cur = 12
    cy_l  = LEGEND_H // 2
    for label, lcolor in legend_items:
        cv2.circle(legend, (x_cur + 7, cy_l), 7,  lcolor,          -1)
        cv2.circle(legend, (x_cur + 7, cy_l), 8,  (255, 255, 255),  1)
        cv2.putText(legend, label, (x_cur + 20, cy_l + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA)
        x_cur += 95

    # Exit symbol in legend
    cv2.circle(legend, (x_cur + 7, cy_l), 8,  (255, 255, 255), -1)
    cv2.circle(legend, (x_cur + 7, cy_l), 7,  COLORS['exit'],  -1)
    cv2.putText(legend, "Exit", (x_cur + 20, cy_l + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA)

    # Fire symbol
    x_cur += 95
    cv2.rectangle(legend, (x_cur + 2, cy_l - 7), (x_cur + 14, cy_l + 7), COLORS['fire'], -1)
    cv2.putText(legend, "Fire", (x_cur + 20, cy_l + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA)

    return np.vstack([banner, frame, legend])


# ---------------------------------------------------------------------------
# Resize helper for video encoding
# ---------------------------------------------------------------------------
def _resize_for_video(frame: np.ndarray, max_width: int = 900) -> np.ndarray:
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame


# ---------------------------------------------------------------------------
# Rollout + frame capture (used by app.py Tab 3)
# ---------------------------------------------------------------------------
def simulate_and_record_rollout(
    floor_key: str = "ground",
    policy: Optional[Union[ActorCriticNetwork, ActorCriticRNN]] = None,
    num_agents: int = 15,
    max_ticks: int = 200,
    seed: int = 42,
    title_suffix: str = "",
) -> List[np.ndarray]:
    """
    Runs a single policy rollout and returns all rendered frames.
    If policy is None, uses random valid actions (untrained baseline).
    """
    img, free_space, graph, exits, _ = load_floor(floor_key)
    lengths, _ = nx.multi_source_dijkstra(graph, sources=exits, weight='weight')
    dist_map   = {graph.nodes[n]['pos']: d for n, d in lengths.items()}
    exit_coords = [graph.nodes[e]['pos'] for e in exits]

    env = HazardEvacEnv(
        free_space_grid=free_space,
        dist_map=dist_map,
        exit_coords=exit_coords,
        num_agents=num_agents,
        max_ticks=max_ticks,
        fire_origin=(250, 580),
        exit_reach_dist=25.0,
        exit_reward=300.0,
    )

    obs, _  = env.reset(seed=seed)
    frames  = []
    agent_trails: Dict[int, List[Tuple[int, int]]] = {aid: [a.pos] for aid, a in env.agents.items()}
    agent_hiddens: Dict[int, torch.Tensor] = {}

    # Convert to BGR once
    base_bgr = cv2.cvtColor(np.asarray(img, dtype=np.uint8), cv2.COLOR_RGB2BGR)

    for _ in range(max_ticks):
        actions: Dict[int, int] = {}
        for aid, o in obs.items():
            s, p, m = o["spatial"], o["proprio"], o["action_mask"]
            if policy is None:
                valid = np.where(m == 1)[0]
                actions[aid] = int(np.random.choice(valid))
            elif isinstance(policy, ActorCriticRNN):
                h_in = agent_hiddens.get(aid)
                act, _, _, h_next = policy.act_rnn(s, p, m, hidden_state=h_in, deterministic=True)
                agent_hiddens[aid] = h_next.detach()
                actions[aid] = act
            else:
                act, _, _ = policy.act(s, p, m, deterministic=True)
                actions[aid] = act

        obs, _, _, _, info = env.step(actions)

        for aid, agent in env.agents.items():
            if not agent.is_evacuated and not agent.is_dead:
                agent_trails[aid].append(agent.pos)

        ep_title = f"Episode Rollout {title_suffix}"
        frame = draw_simulation_frame(base_bgr, env, title=ep_title, agent_trails=agent_trails)
        frames.append(frame)

        if info["all_done"]:
            break

    return frames


# ---------------------------------------------------------------------------
# Side-by-side comparison video (used by app.py Tab 3)
# ---------------------------------------------------------------------------
def generate_comparison_video(
    output_path: str = "outputs/comparison_evacuation.mp4",
    floor_key: str = "ground",
    trained_policy_path: str = "checkpoints/rnn_policy_latest.pt",
    num_agents: int = 15,
    max_ticks: int = 200,
    seed: int = 42,
    fps: int = 12,
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print("[Video] Simulating untrained (random) baseline...")
    frames_untrained = simulate_and_record_rollout(
        floor_key=floor_key, policy=None,
        num_agents=num_agents, max_ticks=max_ticks, seed=seed,
        title_suffix="(Untrained)"
    )

    print("[Video] Simulating trained policy...")
    trained_policy = ActorCriticRNN()
    if os.path.exists(trained_policy_path):
        trained_policy.load(trained_policy_path)
        print(f"[Video] Loaded checkpoint: {trained_policy_path}")
    else:
        print("[Video] Checkpoint not found — using random initialization.")

    frames_trained = simulate_and_record_rollout(
        floor_key=floor_key, policy=trained_policy,
        num_agents=num_agents, max_ticks=max_ticks, seed=seed,
        title_suffix="(Trained RL)"
    )

    # Pad shorter rollout to same length
    n_frames = max(len(frames_untrained), len(frames_trained))
    while len(frames_untrained) < n_frames:
        frames_untrained.append(frames_untrained[-1])
    while len(frames_trained) < n_frames:
        frames_trained.append(frames_trained[-1])

    # Resize both sides to same dimensions
    h = max(frames_untrained[0].shape[0], frames_trained[0].shape[0])
    w = max(frames_untrained[0].shape[1], frames_trained[0].shape[1])
    # Resize to max 800px wide each
    target_w = min(w, 800)
    scale = target_w / w
    target_h = int(h * scale)

    divider = np.full((target_h, 6, 3), 60, dtype=np.uint8)
    comp_w  = target_w * 2 + 6

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (comp_w, target_h))

    print(f"[Video] Encoding {n_frames} frames to {output_path}...")
    for t in range(n_frames):
        left  = cv2.resize(frames_untrained[t], (target_w, target_h), interpolation=cv2.INTER_AREA)
        right = cv2.resize(frames_trained[t],   (target_w, target_h), interpolation=cv2.INTER_AREA)
        combined = np.hstack([left, divider, right])
        writer.write(combined)

    writer.release()
    sz = os.path.getsize(output_path)
    print(f"✅ [Video] Saved: {output_path} ({sz:,} bytes)")
    return output_path
