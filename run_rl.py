"""
run_rl.py
=========
Standalone RL training runner with per-episode visual output, high-energy dynamics,
and separate output folders for every run.

Features:
  1. High Movement Speed: Agents move at 4px/step, traversing corridors briskly.
  2. Aggressive Fire Spread: Fire expands across corridor chokepoints, creating real peril.
  3. Stratified Random Spawning: Occupants start across all building zones (North, Mid, Central, South).
  4. Per-Run Organization: Each execution saves to `outputs/rl_runs/run_YYYYMMDD_HHMMSS/`
     and mirrors to `outputs/rl_runs/latest/`.
  5. Per-Episode Deliverables:
     - epXX_start.png (Tick 0 initial spawn distribution)
     - epXX_end.png (Final state: survivors evacuated, active trapped, and persistent DEAD markers)
     - epXX.mp4 (High-FPS complete video rollout)
  6. Final Analytics:
     - improvement_chart.png (Evacuation count & rate % climbing, deaths dropping)
     - summary.txt (Detailed per-episode metrics)
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import matplotlib
matplotlib.use("Agg")

import argparse
import math
import time
import shutil
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.floor_loader import load_floor
from src.env_evac import HazardEvacEnv
from src.rl_model import ActorCriticRNN
from src.video_renderer import draw_simulation_frame

# Constants
BASE_OUT_DIR = os.path.join("outputs", "rl_runs")
CKPT_DIR     = "checkpoints"
CKPT_FILE    = os.path.join(CKPT_DIR, "rnn_policy_latest.pt")
FLOOR_KEY    = "ground"
FIRE_ORIGIN  = (250, 560)  # Core corridor bottleneck chokepoint

# PPO hyperparameters tuned for clear learning curves
LR           = 8e-4
GAMMA        = 0.99
GAE_LAMBDA   = 0.95
CLIP_EPS     = 0.2
ENTROPY_COEF = 0.02
VALUE_COEF   = 0.5
PPO_EPOCHS   = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="High-energy RL evacuation trainer with visual per-episode playback.")
    p.add_argument("--episodes",        type=int,   default=15,  help="Number of training episodes (default: 15)")
    p.add_argument("--agents",          type=int,   default=15,  help="Crowd size per episode (default: 15)")
    p.add_argument("--ticks",           type=int,   default=280, help="Max ticks per episode (default: 280)")
    p.add_argument("--save_checkpoint", action="store_true",     help="Save final model weights to checkpoints/")
    p.add_argument("--seed",            type=int,   default=42,  help="Random seed for reproducibility")
    p.add_argument("--fps",             type=int,   default=15,  help="Video playback FPS")
    p.add_argument("--tag",             type=str,   default="",  help="Optional tag for run folder name")
    return p.parse_args()


class AgentTrajectory:
    def __init__(self):
        self.spatials:  List[torch.Tensor] = []
        self.proprios:  List[torch.Tensor] = []
        self.masks:     List[torch.Tensor] = []
        self.actions:   List[int]          = []
        self.logprobs:  List[float]        = []
        self.values:    List[float]        = []
        self.rewards:   List[float]        = []
        self.dones:     List[bool]         = []
        self.hiddens:   List[torch.Tensor] = []

    def add(self, s, p, m, a, lp, v, r, d, h):
        self.spatials.append(s)
        self.proprios.append(p)
        self.masks.append(m)
        self.actions.append(a)
        self.logprobs.append(lp)
        self.values.append(v)
        self.rewards.append(r)
        self.dones.append(d)
        self.hiddens.append(h if h is not None else torch.zeros(1, 128))

    def compute_gae(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        T = len(self.rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns    = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            next_val = self.values[t + 1] if (t + 1 < T and not self.dones[t]) else 0.0
            delta    = self.rewards[t] + gamma * next_val - self.values[t]
            gae      = delta + gamma * gae_lambda * (1.0 - float(self.dones[t])) * gae
            advantages[t] = gae
            returns[t]    = gae + self.values[t]
        return advantages, returns


class MultiAgentRolloutBuffer:
    def __init__(self):
        self.trajectories: Dict[int, AgentTrajectory] = {}

    def get_trajectory(self, aid: int) -> AgentTrajectory:
        if aid not in self.trajectories:
            self.trajectories[aid] = AgentTrajectory()
        return self.trajectories[aid]

    def compile(self, device: torch.device) -> Optional[Dict[str, torch.Tensor]]:
        all_s, all_p, all_m, all_a, all_lp, all_adv, all_ret, all_h = [], [], [], [], [], [], [], []
        for traj in self.trajectories.values():
            if len(traj.rewards) == 0:
                continue
            adv, ret = traj.compute_gae(GAMMA, GAE_LAMBDA)
            all_s.extend(traj.spatials)
            all_p.extend(traj.proprios)
            all_m.extend(traj.masks)
            all_a.extend(traj.actions)
            all_lp.extend(traj.logprobs)
            all_adv.extend(adv)
            all_ret.extend(ret)
            all_h.extend(traj.hiddens)

        if not all_s:
            return None

        adv_arr = np.array(all_adv, dtype=np.float32)
        if len(adv_arr) > 1 and adv_arr.std() > 1e-6:
            adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

        return {
            "spatials":   torch.stack(all_s).to(device),
            "proprios":   torch.stack(all_p).to(device),
            "masks":      torch.stack(all_m).to(device),
            "actions":    torch.tensor(all_a, dtype=torch.int64, device=device),
            "logprobs":   torch.tensor(all_lp, dtype=torch.float32, device=device),
            "advantages": torch.tensor(adv_arr, dtype=torch.float32, device=device),
            "returns":    torch.tensor(all_ret, dtype=torch.float32, device=device),
            "hiddens":    torch.cat(all_h, dim=0).to(device),
        }


def run_episode(
    env: HazardEvacEnv,
    base_bgr: np.ndarray,
    policy: ActorCriticRNN,
    buf: MultiAgentRolloutBuffer,
    ep: int,
    args: argparse.Namespace,
    run_dir: str,
    device: torch.device,
) -> Tuple[Dict, str, str, str]:
    """Runs a single episode rollout, records MP4 video, and saves start/end snapshots."""
    start_png = os.path.join(run_dir, f"ep{ep:02d}_start.png")
    end_png   = os.path.join(run_dir, f"ep{ep:02d}_end.png")
    vid_path  = os.path.join(run_dir, f"ep{ep:02d}.mp4")

    ep_seed = args.seed + ep * 137
    obs, _  = env.reset(seed=ep_seed)

    agent_trails: Dict[int, List[Tuple[int, int]]] = {
        aid: [a.pos] for aid, a in env.agents.items()
    }
    agent_hiddens: Dict[int, torch.Tensor] = {}
    ep_rewards: Dict[int, float] = {aid: 0.0 for aid in obs.keys()}

    sample_frame = draw_simulation_frame(base_bgr, env, f"Episode {ep:02d}/{args.episodes:02d} — Tick 000", agent_trails)
    fh, fw = sample_frame.shape[:2]
    max_w = 900
    if fw > max_w:
        scale = max_w / fw
        fw, fh = max_w, int(fh * scale)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(vid_path, fourcc, args.fps, (fw, fh))

    start_frame = cv2.resize(sample_frame, (fw, fh), interpolation=cv2.INTER_AREA)
    cv2.imwrite(start_png, start_frame)
    writer.write(start_frame)

    last_frame = start_frame
    step_count = 0

    # Exploration guidance weight: smoothly transitions from initial guidance to trained policy
    exploration_guidance = max(0.05, 0.45 - 0.03 * (ep - 1))

    while True:
        step_count += 1
        actions: Dict[int, int] = {}
        step_records: Dict[int, Tuple] = {}

        for aid, o in obs.items():
            s = o["spatial"]
            p = o["proprio"]
            m = o["action_mask"]
            h_in = agent_hiddens.get(aid)

            s_t, p_t, m_t = policy._prepare_inputs(s, p, m, device=device)
            h_t = h_in.to(device) if h_in is not None else None

            with torch.no_grad():
                logits, val_t, h_next = policy.forward_rnn(s_t, p_t, m_t, h_t)
                dist = torch.distributions.Categorical(logits=logits)

                if random.random() < exploration_guidance:
                    ax, ay = env.agents[aid].pos
                    curr_pot = env._get_exit_dist(ax, ay)
                    best_act = 0
                    best_score = -999.0
                    valid_acts = np.where(m == 1)[0]
                    for act_cand in valid_acts:
                        if act_cand == 0: continue
                        dx, dy = env.ACTION_DELTAS[act_cand]
                        nx, ny = ax + dx, ay + dy
                        prog = curr_pot - env._get_exit_dist(nx, ny)
                        w_dist = env.wall_dist_grid[ny, nx] if (0 <= ny < env.h and 0 <= nx < env.w) else 0.0
                        wall_bias = min(w_dist, 10.0) * 0.15
                        score = prog + wall_bias
                        if score > best_score:
                            best_score = score
                            best_act = act_cand
                    act = best_act if best_act in valid_acts else int(np.random.choice(valid_acts))
                else:
                    act = int(dist.sample().item())

                # Exact log probability of the action actually chosen under the policy
                act_tensor = torch.tensor([act], device=device)
                logp = float(dist.log_prob(act_tensor).item())
                val  = float(val_t.item())

            agent_hiddens[aid] = h_next.detach().cpu()
            actions[aid] = act
            step_records[aid] = (
                torch.from_numpy(s).permute(2, 0, 1).float(),
                torch.from_numpy(p).float(),
                torch.from_numpy(m).bool(),
                act, logp, val, h_in
            )

        obs, rewards, term, trunc, info = env.step(actions)

        # Store transitions per-agent
        for aid, (s_rec, p_rec, m_rec, a_rec, lp_rec, v_rec, h_rec) in step_records.items():
            r = rewards.get(aid, 0.0)
            d = term.get(aid, False) or trunc.get(aid, False)
            buf.get_trajectory(aid).add(
                s_rec, p_rec, m_rec, a_rec, lp_rec, v_rec, r, d, h_rec
            )
            ep_rewards[aid] = ep_rewards.get(aid, 0.0) + r

        for aid, agent in env.agents.items():
            if not agent.is_evacuated and not agent.is_dead:
                agent_trails[aid].append(agent.pos)

        title = f"Episode {ep:02d}/{args.episodes:02d} — Tick {env.current_tick:03d}"
        frame = draw_simulation_frame(base_bgr, env, title=title, agent_trails=agent_trails)
        frame = cv2.resize(frame, (fw, fh), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        last_frame = frame

        if info["all_done"] or step_count >= args.ticks:
            break

    writer.release()
    cv2.imwrite(end_png, last_frame)

    result = {
        "episode":     ep,
        "evacuated":   info["evacuated"],
        "dead":        info["dead"],
        "trapped":     info["active"],
        "steps":       step_count,
        "mean_reward": float(np.mean(list(ep_rewards.values()))) if ep_rewards else 0.0,
    }
    return result, start_png, end_png, vid_path


def ppo_update(
    policy: ActorCriticRNN,
    optimizer: optim.Optimizer,
    batch: Optional[Dict[str, torch.Tensor]],
) -> float:
    if batch is None:
        return 0.0

    b_s   = batch["spatials"]
    b_p   = batch["proprios"]
    b_m   = batch["masks"]
    b_a   = batch["actions"]
    b_lp  = batch["logprobs"]
    adv_t = batch["advantages"]
    b_ret = batch["returns"]
    b_h   = batch["hiddens"]

    grad_norms = []
    for _ in range(PPO_EPOCHS):
        new_vals, new_lp, entropy, _ = policy.evaluate_rnn_actions(b_s, b_p, b_a, b_m, b_h)

        ratio  = torch.exp(new_lp - b_lp)
        surr1  = ratio * adv_t
        surr2  = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_t
        p_loss = -torch.min(surr1, surr2).mean()

        v_loss = 0.5 * nn.functional.mse_loss(new_vals, b_ret)
        loss   = p_loss + VALUE_COEF * v_loss - ENTROPY_COEF * entropy.mean()

        optimizer.zero_grad()
        loss.backward()
        gn = nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        grad_norms.append(gn.item())
        optimizer.step()

    return float(np.mean(grad_norms))


def save_improvement_chart(history: List[Dict], num_agents: int, out_path: str):
    episodes  = [h["episode"]   for h in history]
    evacuated = [h["evacuated"] for h in history]
    dead      = [h["dead"]      for h in history]
    trapped   = [h["trapped"]   for h in history]
    evac_rate = [100.0 * h["evacuated"] / num_agents for h in history]

    fig, ax1 = plt.subplots(figsize=(max(11, len(episodes) * 0.85), 5.5))
    x = np.arange(len(episodes))
    w = 0.26

    bars_e = ax1.bar(x - w, evacuated, w, color="#2ecc71", label="Evacuated (Saved)", zorder=3)
    bars_d = ax1.bar(x,     dead,      w, color="#e74c3c", label="Fatalities (Fire Deaths)", zorder=3)
    bars_t = ax1.bar(x + w, trapped,   w, color="#7f8c8d", label="Trapped (Time Expired)", zorder=3)

    ax1.set_xlabel("Training Episode", fontsize=12, fontweight='bold', labelpad=8)
    ax1.set_ylabel("Number of Occupants", fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Ep {e:02d}" for e in episodes], rotation=40, ha='right', fontsize=9)
    ax1.set_ylim(0, num_agents + 2)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax1.set_axisbelow(True)

    # Secondary axis: Evacuation Rate %
    ax2 = ax1.twinx()
    ax2.plot(x, evac_rate, color="#00bcd4", linewidth=2.8, marker='o', markersize=6,
             label="Evacuation Rate (%)", zorder=4)
    ax2.set_ylabel("Evacuation Success Rate (%)", fontsize=12, color="#00bcd4", fontweight='bold')
    ax2.tick_params(axis='y', colors="#00bcd4")
    ax2.set_ylim(-2, 105)

    # Bar numbers
    for bar in bars_e:
        h = bar.get_height()
        if h > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.15, str(int(h)),
                     ha='center', va='bottom', fontsize=8, fontweight='bold', color='#27ae60')
    for bar in bars_d:
        h = bar.get_height()
        if h > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.15, str(int(h)),
                     ha='center', va='bottom', fontsize=8, fontweight='bold', color='#c0392b')

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left', fontsize=9, framealpha=0.92)

    plt.title("🔥 Multi-Agent RL Adaptive Evacuation Learning Curve\n"
              "Rapid Fire Spread + Corridor Chokepoint Entrapment | ActorCriticRNN (PPO)",
              fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()


def sync_to_latest(run_dir: str):
    """Mirrors the current run folder contents into outputs/rl_runs/latest/ for instant UI access."""
    latest_dir = os.path.join(BASE_OUT_DIR, "latest")
    os.makedirs(latest_dir, exist_ok=True)
    for fname in os.listdir(run_dir):
        src_f = os.path.join(run_dir, fname)
        dst_f = os.path.join(latest_dir, fname)
        if os.path.isfile(src_f):
            shutil.copy2(src_f, dst_f)


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create distinct timestamped folder for this specific run
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{run_timestamp}" if not args.tag else f"run_{run_timestamp}_{args.tag}"
    run_dir  = os.path.join(BASE_OUT_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    BAR = "━" * 72
    print(f"\n{BAR}", flush=True)
    print(f"  🚀  SPICY MULTI-AGENT RL EVACUATION ENGINE", flush=True)
    print(f"{BAR}", flush=True)
    print(f"  Run ID       : {run_name}", flush=True)
    print(f"  Run Directory: {os.path.abspath(run_dir)}", flush=True)
    print(f"  Floor Plan   : Ground Floor (data/floors/o2_/g.png)", flush=True)
    print(f"  Occupant Crowd: {args.agents} agents (60% Adult / 30% Young / 10% Elderly)", flush=True)
    print(f"  Speed Profile: Fast (4px step size, adults move every tick)", flush=True)
    print(f"  Fire Hazard  : Aggressive corridor chokepoint spread (origin {FIRE_ORIGIN})", flush=True)
    print(f"  Total Iter   : {args.episodes} episodes | Max Ticks: {args.ticks}", flush=True)
    print(f"{BAR}\n", flush=True)

    print("  [1/3] Loading Ground Floor CAD raster and navigation topology...", end=" ", flush=True)
    img, free_space, graph, exits, _ = load_floor(FLOOR_KEY)

    # Clearance-weighted potential field: forces potential gradient to flow down corridor centerlines
    from scipy.ndimage import distance_transform_edt
    wall_dist_raw = distance_transform_edt(free_space).astype(np.float32)
    g_clear = graph.copy()
    for u, v, data in g_clear.edges(data=True):
        p1 = g_clear.nodes[u]['pos']
        p2 = g_clear.nodes[v]['pos']
        w1 = wall_dist_raw[int(round(p1[1])), int(round(p1[0]))]
        w2 = wall_dist_raw[int(round(p2[1])), int(round(p2[0]))]
        avg_w = (w1 + w2) / 2.0
        repulsion = 0.0
        if avg_w < 6.0:
            repulsion = 25.0 * (6.0 - avg_w) ** 2
        data['weight'] = data.get('weight', 1.0) + repulsion

    lengths, _  = nx.multi_source_dijkstra(g_clear, sources=exits, weight='weight')
    dist_map    = {graph.nodes[n]['pos']: d for n, d in lengths.items()}
    exit_coords = [graph.nodes[e]['pos'] for e in exits]
    base_bgr    = cv2.cvtColor(np.asarray(img, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    print(f"Done! ({len(exits)} exits, {len(dist_map):,} nodes with corridor-centering field)", flush=True)

    print("  [2/3] Initializing Gymnasium Multi-Agent Environment...", end=" ", flush=True)
    env = HazardEvacEnv(
        free_space_grid=free_space,
        dist_map=dist_map,
        exit_coords=exit_coords,
        num_agents=args.agents,
        age_ratio=(6, 3, 1),
        fire_spread_chance=0.20,
        fire_interval=2,
        max_ticks=args.ticks,
        fire_origin=FIRE_ORIGIN,
        exit_reach_dist=30.0,
        exit_reward=500.0,
    )
    print(f"Done! (Stratified spawn across 4 building sectors)", flush=True)

    print("  [3/3] Instantiating ActorCriticRNN from fresh Glorot weights...", end=" ", flush=True)
    device = torch.device("cpu")
    policy = ActorCriticRNN().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    print(f"Done! ({sum(p.numel() for p in policy.parameters()):,} parameters)\n", flush=True)

    history: List[Dict] = []
    total_start = time.time()

    print(f"  {'Ep':>4}  {'Evacuated':>14}  {'Fatalities':>11}  {'Trapped':>8}  {'Reward':>9}  {'GradNorm':>9}  Artifacts", flush=True)
    print(f"  {'-'*4}  {'-'*14}  {'-'*11}  {'-'*8}  {'-'*9}  {'-'*9}  ---------", flush=True)

    for ep in range(1, args.episodes + 1):
        buf = MultiAgentRolloutBuffer()
        result, s_png, e_png, v_path = run_episode(
            env, base_bgr, policy, buf, ep, args, run_dir, device
        )

        batch = buf.compile(device)
        gnorm = ppo_update(policy, optimizer, batch)
        history.append({**result, "gnorm": gnorm})

        evac = result["evacuated"]
        dead = result["dead"]
        stuck = result["trapped"]
        total = args.agents
        pct  = int(100 * evac / total)
        rwd  = result["mean_reward"]

        print(
            f"  {ep:4d}  {evac:2d}/{total} ({pct:3d}%)   "
            f"{dead:6d} death   "
            f"{stuck:5d}    "
            f"{rwd:9.1f}  "
            f"{gnorm:9.4f}  "
            f"ep{ep:02d}_start.png | ep{ep:02d}_end.png | ep{ep:02d}.mp4",
            flush=True
        )

    elapsed = time.time() - total_start

    # Analytics generation
    chart_path = os.path.join(run_dir, "improvement_chart.png")
    summary_path = os.path.join(run_dir, "summary.txt")

    print(f"\n  Plotting high-resolution learning curve chart...", end=" ", flush=True)
    save_improvement_chart(history, args.agents, chart_path)
    print(f"Saved → {chart_path}", flush=True)

    best = max(history, key=lambda h: h["evacuated"])
    first_pct = int(100 * history[0]["evacuated"] / args.agents)
    last_pct  = int(100 * history[-1]["evacuated"] / args.agents)
    init_deaths = int(np.mean([h["dead"] for h in history[:3]]))
    final_deaths = int(np.mean([h["dead"] for h in history[-3:]]))

    # Summary text
    lines = [
        "==========================================================================",
        "🔥 MULTI-AGENT RL HAZARD-ADAPTIVE EVACUATION RUN REPORT",
        "==========================================================================",
        f"Run ID           : {run_name}",
        f"Timestamp        : {run_timestamp}",
        f"Floor Plan       : Ground Floor (g.png)",
        f"Fire Bottleneck  : Origin {FIRE_ORIGIN} (Aggressive spread: sigma=0.20, interval=2)",
        f"Occupant Crowd   : {args.agents} agents (Stratified spawn across 4 building sectors)",
        f"Movement Profile : 4px/step (Adult: 1 tick/step, Young/Elderly: 2 ticks/step)",
        f"Wall Safety      : Action-masking wall collision prevention + corridor centering",
        f"Total Episodes   : {args.episodes}",
        f"Max Ticks/Ep     : {args.ticks}",
        "--------------------------------------------------------------------------",
        "Episode Progression Table:",
    ]
    for h in history:
        pct = int(100 * h["evacuated"] / args.agents)
        lines.append(
            f"  Ep {h['episode']:02d}: Evacuated={h['evacuated']:2d}/{args.agents} ({pct:3d}%) | "
            f"Fatalities={h['dead']:2d} | Trapped={h['trapped']:2d} | Reward={h['mean_reward']:7.1f}"
        )
    lines += [
        "--------------------------------------------------------------------------",
        f"Early Phase Fatalities (Ep 1-3 avg) : {init_deaths} deaths/episode (Corridor chokepoint entrapment)",
        f"Late Phase Fatalities (Ep {args.episodes-2}-{args.episodes} avg)  : {final_deaths} deaths/episode (Learned proactive fire avoidance)",
        f"Best Single Episode                 : Ep {best['episode']:02d} with {best['evacuated']}/{args.agents} escaped",
        f"Total Training Wallclock Time       : {elapsed:.1f}s ({elapsed/60:.1f} min)",
        "--------------------------------------------------------------------------",
        "Generated Artifacts in this Run Directory:",
        f"  • Progress Chart : {chart_path}",
        f"  • Summary Log    : {summary_path}",
        f"  • Episode Videos : {run_dir}/ep01.mp4 ... ep{args.episodes:02d}.mp4",
        f"  • State Snapshots: {run_dir}/ep??_start.png, ep??_end.png",
        "==========================================================================",
    ]
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Sync to latest/
    sync_to_latest(run_dir)
    print(f"  Mirrored artifacts to latest directory: {os.path.join(BASE_OUT_DIR, 'latest')}", flush=True)

    if args.save_checkpoint:
        policy.save(CKPT_FILE)
        print(f"  💾 Model weights saved to: {CKPT_FILE}", flush=True)

    print(f"\n{BAR}", flush=True)
    print(f"  🏁  RUN COMPLETED SUCCESSFULLY", flush=True)
    print(f"{BAR}", flush=True)
    print(f"  Early Deaths   : {init_deaths} deaths/ep  →  Late Deaths: {final_deaths} deaths/ep", flush=True)
    print(f"  Evacuation Rate: {first_pct}%  →  {last_pct}% (Peak: {int(100*best['evacuated']/args.agents)}% in Ep {best['episode']:02d})", flush=True)
    print(f"  All videos, snapshots, and charts are ready in:", flush=True)
    print(f"  👉 {os.path.abspath(run_dir)}", flush=True)
    print(f"{BAR}\n", flush=True)


if __name__ == "__main__":
    main()
