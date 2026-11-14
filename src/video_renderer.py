import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import networkx as nx

from src.floor_loader import load_floor
from src.env_evac import HazardEvacEnv
from src.rl_model import ActorCriticNetwork, ActorCriticRNN


def draw_simulation_frame(
    base_img: np.ndarray,
    env: HazardEvacEnv,
    title_suffix: str = "",
    agent_trails: Optional[Dict[int, List[Tuple[int, int]]]] = None
) -> np.ndarray:
    """Renders a single simulation frame with agents, fire, and metrics."""
    frame = base_img.copy()
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    elif frame.shape[2] == 3 and not base_img.flags.writeable:
        frame = frame.copy()

    # 1. Overlay Fire Heat / Smoke Buffer
    heat_mask = (env.fire_dist_grid > 0.0) & (env.fire_dist_grid < 0.25) & ~env.fire_grid
    frame[heat_mask] = (frame[heat_mask] * 0.5 + np.array([30, 100, 255]) * 0.5).astype(np.uint8)

    # 2. Overlay Active Fire Front
    frame[env.fire_grid] = [20, 40, 240] # Bright Fire Red (BGR)

    # 3. Draw Exits
    for ex, ey in env.exit_coords:
        cv2.circle(frame, (int(ex), int(ey)), 9, (0, 220, 0), -1)
        cv2.circle(frame, (int(ex), int(ey)), 11, (255, 255, 255), 2)

    # 4. Draw Trails (if provided)
    if agent_trails:
        for aid, trail in agent_trails.items():
            if len(trail) >= 2:
                for k in range(max(0, len(trail) - 8), len(trail) - 1):
                    p1, p2 = trail[k], trail[k + 1]
                    cv2.line(frame, p1, p2, (200, 200, 200), 1, cv2.LINE_AA)

    # 5. Draw Agents
    # Colors by demographic: Adult=Green, Young=Cyan/Blue, Elderly=Magenta
    demo_colors = {
        'adult': (0, 220, 50),
        'young': (255, 180, 0),
        'elderly': (220, 20, 220)
    }

    for aid, agent in env.agents.items():
        cx, cy = int(agent.pos[0]), int(agent.pos[1])
        if agent.is_dead:
            # Red/Black cross
            cv2.line(frame, (cx - 4, cy - 4), (cx + 4, cy + 4), (0, 0, 0), 2)
            cv2.line(frame, (cx - 4, cy + 4), (cx + 4, cy - 4), (0, 0, 0), 2)
        elif not agent.is_evacuated:
            color = demo_colors.get(agent.agent_type, (0, 255, 0))
            cv2.circle(frame, (cx, cy), 4, color, -1)
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), 1)

    # 6. Top Status Banner
    banner_h = 35
    banner = np.zeros((banner_h, frame.shape[1], 3), dtype=np.uint8)
    active_cnt = sum(1 for a in env.agents.values() if not a.is_evacuated and not a.is_dead)
    evac_cnt = sum(1 for a in env.agents.values() if a.is_evacuated)
    dead_cnt = sum(1 for a in env.agents.values() if a.is_dead)

    text = f"Tick: {env.current_tick:03d} | Active: {active_cnt:02d} | Evac: {evac_cnt:02d} | Dead: {dead_cnt:02d} {title_suffix}"
    cv2.putText(banner, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([banner, frame])


def simulate_and_record_rollout(
    floor_key: str = "ground",
    policy: Optional[Union[ActorCriticNetwork, ActorCriticRNN]] = None,
    num_agents: int = 20,
    max_ticks: int = 120,
    seed: int = 42,
    title_suffix: str = ""
) -> List[np.ndarray]:
    """Runs a single rollout and captures simulation frames."""
    img, free_space, graph, exits, _ = load_floor(floor_key)
    lengths, _ = nx.multi_source_dijkstra(graph, sources=exits, weight='weight')
    dist_map = {graph.nodes[n]['pos']: d for n, d in lengths.items()}
    exit_coords = [graph.nodes[e]['pos'] for e in exits]

    env = HazardEvacEnv(
        free_space_grid=free_space,
        dist_map=dist_map,
        exit_coords=exit_coords,
        num_agents=num_agents,
        max_ticks=max_ticks,
        fire_origin=(250, 580)
    )

    obs, info = env.reset(seed=seed)
    frames = []
    agent_trails: Dict[int, List[Tuple[int, int]]] = {aid: [a.pos] for aid, a in env.agents.items()}
    agent_hiddens: Dict[int, torch.Tensor] = {}

    for tick in range(max_ticks):
        actions = {}
        for aid, o in obs.items():
            s = o["spatial"]
            p = o["proprio"]
            m = o["action_mask"]

            if policy is None:
                # Random valid move
                valid_actions = np.where(m == 1)[0]
                actions[aid] = int(np.random.choice(valid_actions))
            elif isinstance(policy, ActorCriticRNN):
                h_in = agent_hiddens.get(aid, None)
                act, _, _, h_next = policy.act_rnn(s, p, m, hidden_state=h_in, deterministic=True)
                agent_hiddens[aid] = h_next.detach()
                actions[aid] = act
            else:
                act, _, _ = policy.act(s, p, m, deterministic=True)
                actions[aid] = act

        obs, rewards, term, trunc, info = env.step(actions)

        for aid, a in env.agents.items():
            if not a.is_evacuated:
                agent_trails[aid].append(a.pos)

        frame = draw_simulation_frame(img, env, title_suffix=title_suffix, agent_trails=agent_trails)
        frames.append(frame)

        if info["all_done"]:
            break

    return frames


def generate_comparison_video(
    output_path: str = "outputs/comparison_evacuation.mp4",
    floor_key: str = "ground",
    trained_policy_path: str = "checkpoints/rnn_policy_latest.pt",
    num_agents: int = 20,
    max_ticks: int = 120,
    seed: int = 42,
    fps: int = 10
) -> str:
    """Generates side-by-side MP4 comparing Untrained Policy vs Trained Policy."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # 1. Untrained Rollout
    print("[Video] Simulating Untrained (Episode 0) Baseline...")
    frames_untrained = simulate_and_record_rollout(
        floor_key=floor_key,
        policy=None, # Random untrained
        num_agents=num_agents,
        max_ticks=max_ticks,
        seed=seed,
        title_suffix="(Untrained)"
    )

    # 2. Trained Rollout
    print("[Video] Simulating Trained Policy...")
    trained_policy = ActorCriticRNN()
    if os.path.exists(trained_policy_path):
        trained_policy.load(trained_policy_path)
        print(f"[Video] Loaded checkpoint: {trained_policy_path}")
    else:
        print("[Video] Checkpoint not found; running fresh policy.")

    frames_trained = simulate_and_record_rollout(
        floor_key=floor_key,
        policy=trained_policy,
        num_agents=num_agents,
        max_ticks=max_ticks,
        seed=seed,
        title_suffix="(Trained RL)"
    )

    # Align frame lengths
    n_frames = max(len(frames_untrained), len(frames_trained))
    while len(frames_untrained) < n_frames:
        frames_untrained.append(frames_untrained[-1])
    while len(frames_trained) < n_frames:
        frames_trained.append(frames_trained[-1])

    # 3. Encode Side-by-Side Video
    h, w, _ = frames_untrained[0].shape
    divider = np.full((h, 6, 3), 80, dtype=np.uint8) # Dark grey vertical divider
    comp_w = w * 2 + 6

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (comp_w, h))

    print(f"[Video] Encoding {n_frames} side-by-side frames to {output_path}...")
    for t in range(n_frames):
        left = frames_untrained[t]
        right = frames_trained[t]
        combined = np.hstack([left, divider, right])
        writer.write(combined)

    writer.release()
    print(f"✅ [Video] Comparison video successfully saved: {output_path} ({os.path.getsize(output_path):,} bytes)")
    return output_path
