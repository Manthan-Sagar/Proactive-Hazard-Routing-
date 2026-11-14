"""
run_rl.py
=========
Standalone RL training runner with per-episode visual output.

What this script does:
  1. Loads the Ground Floor (g.png + g.xml) — fixed, no switching needed.
  2. Trains an ActorCriticRNN with PPO from scratch.
  3. Uses curriculum learning: agents start near exits in early episodes,
     gradually expanding spawn radius so the network learns to navigate
     from farther away as training progresses.
  4. After EACH episode saves:
       outputs/rl_runs/ep{N:02d}_start.png  — where agents spawned (tick 0)
       outputs/rl_runs/ep{N:02d}_end.png    — final state (who evacuated / died / stuck)
       outputs/rl_runs/ep{N:02d}.mp4        — full video of agents moving
  5. After ALL episodes saves:
       outputs/rl_runs/improvement_chart.png — bar chart: evacuation rate rising per episode
       outputs/rl_runs/summary.txt           — text summary

Usage:
  python run_rl.py                        # 15 episodes, 15 agents, 300 ticks
  python run_rl.py --episodes 20          # more episodes
  python run_rl.py --agents 20 --ticks 250
  python run_rl.py --episodes 20 --save_checkpoint    # keep final weights

Controls:
  --episodes N      How many training episodes (default 15)
  --agents M        Number of agents per episode (default 15)
  --ticks T         Max simulation ticks per episode (default 300)
  --save_checkpoint Save final weights to checkpoints/rnn_policy_latest.pt
  --no_curriculum   Disable curriculum (spawn anywhere from ep 1 — harder)
  --seed S          Random seed (default 42)
  --fps F           Video frames-per-second (default 15)
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import matplotlib
matplotlib.use("Agg")

import argparse
import math
import time
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple

# Ensure we find the src/ package relative to this script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.floor_loader import load_floor
from src.env_evac import HazardEvacEnv
from src.rl_model import ActorCriticRNN
from src.video_renderer import draw_simulation_frame


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUT_DIR    = os.path.join("outputs", "rl_runs")
CKPT_DIR   = "checkpoints"
CKPT_FILE  = os.path.join(CKPT_DIR, "rnn_policy_latest.pt")
FLOOR_KEY  = "ground"
FIRE_ORIGIN = (250, 580)   # corridor chokepoint (the scenario being studied)

# PPO hyperparameters (tuned for visible improvement in 15 episodes)
LR           = 1e-3     # higher than default for faster early learning
GAMMA        = 0.99
GAE_LAMBDA   = 0.95
CLIP_EPS     = 0.2
ENTROPY_COEF = 0.02
VALUE_COEF   = 0.5
PPO_EPOCHS   = 6        # more optimization passes per rollout

# Curriculum schedule
CURRICULUM_NEAR   = 180   # first 1/3 of episodes: spawn within 180px of exits
CURRICULUM_MID    = 360   # middle 1/3: spawn within 360px
# final 1/3: no restriction (spawn anywhere on navigable graph)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run RL evacuation training with per-episode visual output."
    )
    p.add_argument("--episodes",        type=int,   default=15,  help="Number of training episodes")
    p.add_argument("--agents",          type=int,   default=15,  help="Agents per episode")
    p.add_argument("--ticks",           type=int,   default=300, help="Max ticks per episode")
    p.add_argument("--save_checkpoint", action="store_true",     help="Save final weights to disk")
    p.add_argument("--no_curriculum",   action="store_true",     help="Disable curriculum learning")
    p.add_argument("--seed",            type=int,   default=42,  help="Global random seed")
    p.add_argument("--fps",             type=int,   default=15,  help="Video frames per second")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Curriculum helper
# ---------------------------------------------------------------------------
def get_curriculum_dist(episode: int, total_episodes: int, enabled: bool) -> Optional[float]:
    """Returns the spawn-radius limit for this episode (None = no limit)."""
    if not enabled:
        return None
    phase1_end = total_episodes // 3
    phase2_end = 2 * total_episodes // 3
    if episode <= phase1_end:
        return float(CURRICULUM_NEAR)
    elif episode <= phase2_end:
        return float(CURRICULUM_MID)
    else:
        return None


# ---------------------------------------------------------------------------
# PPO rollout buffer
# ---------------------------------------------------------------------------
class RolloutBuffer:
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

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


# ---------------------------------------------------------------------------
# Single episode rollout + video generation
# ---------------------------------------------------------------------------
def run_episode(
    env: HazardEvacEnv,
    base_bgr: np.ndarray,
    policy: ActorCriticRNN,
    buf: RolloutBuffer,
    ep: int,
    args: argparse.Namespace,
    curriculum_dist: Optional[float],
) -> Tuple[Dict, str, str, str]:
    """
    Runs one full episode rollout:
      - Collects (s,a,r,...) transitions into buf
      - Records frames to a per-episode .mp4
      - Saves start/end PNG snapshots
    Returns: (info_dict, start_png_path, end_png_path, video_path)
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    start_png = os.path.join(OUT_DIR, f"ep{ep:02d}_start.png")
    end_png   = os.path.join(OUT_DIR, f"ep{ep:02d}_end.png")
    vid_path  = os.path.join(OUT_DIR, f"ep{ep:02d}.mp4")

    ep_seed = args.seed + ep * 137
    obs, _  = env.reset(seed=ep_seed, options={"curriculum_max_dist": curriculum_dist})

    agent_trails: Dict[int, List[Tuple[int, int]]] = {
        aid: [a.pos] for aid, a in env.agents.items()
    }
    agent_hiddens: Dict[int, torch.Tensor] = {}
    ep_rewards: Dict[int, float] = {aid: 0.0 for aid in obs.keys()}

    # Open video writer (use first frame dimensions after setup)
    sample_frame = draw_simulation_frame(base_bgr, env, f"Episode {ep}/{args.episodes} — Tick 000", agent_trails)
    fh, fw = sample_frame.shape[:2]
    # Resize if too wide
    max_w = 900
    if fw > max_w:
        scale = max_w / fw
        fw, fh = max_w, int(fh * scale)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(vid_path, fourcc, args.fps, (fw, fh))

    # Save start snapshot (tick 0)
    start_frame = cv2.resize(sample_frame, (fw, fh), interpolation=cv2.INTER_AREA)
    cv2.imwrite(start_png, start_frame)
    writer.write(start_frame)

    last_frame = start_frame
    step_count = 0

    while True:
        step_count += 1
        actions: Dict[int, int] = {}

        for aid, o in obs.items():
            s = o["spatial"]
            p = o["proprio"]
            m = o["action_mask"]
            h_in = agent_hiddens.get(aid)

            act, logp, val, h_next = policy.act_rnn(s, p, m, hidden_state=h_in)
            agent_hiddens[aid] = h_next.detach()
            actions[aid] = act

            buf.add(
                torch.from_numpy(s).permute(2, 0, 1).float(),
                torch.from_numpy(p).float(),
                torch.from_numpy(m).bool(),
                act, logp, val,
                0.0,   # reward filled in after step
                False, # done filled in after step
                h_in,
            )

        obs, rewards, term, trunc, info = env.step(actions)

        # Fill in rewards/dones for the transitions we just added
        active_aids = list(actions.keys())
        for i, aid in enumerate(active_aids):
            r = rewards.get(aid, 0.0)
            d = term.get(aid, False) or trunc.get(aid, False)
            idx = len(buf) - len(active_aids) + i
            buf.rewards[idx] = r
            buf.dones[idx]   = d
            ep_rewards[aid]  = ep_rewards.get(aid, 0.0) + r

        # Update trails
        for aid, agent in env.agents.items():
            if not agent.is_evacuated and not agent.is_dead:
                agent_trails[aid].append(agent.pos)

        # Render and write frame
        title = f"Episode {ep}/{args.episodes} — Tick {env.current_tick:03d}"
        frame = draw_simulation_frame(base_bgr, env, title=title, agent_trails=agent_trails)
        frame = cv2.resize(frame, (fw, fh), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        last_frame = frame

        if info["all_done"] or step_count >= args.ticks:
            break

    writer.release()

    # Save end snapshot (final state)
    cv2.imwrite(end_png, last_frame)

    result = {
        "episode":   ep,
        "evacuated": info["evacuated"],
        "dead":      info["dead"],
        "trapped":   info["active"],   # still in building at end
        "steps":     step_count,
        "mean_reward": float(np.mean(list(ep_rewards.values()))) if ep_rewards else 0.0,
    }
    return result, start_png, end_png, vid_path


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------
def ppo_update(
    policy: ActorCriticRNN,
    optimizer: optim.Optimizer,
    buf: RolloutBuffer,
    device: torch.device,
) -> float:
    """Runs PPO_EPOCHS gradient steps. Returns mean gradient norm."""
    T = len(buf)
    if T == 0:
        return 0.0

    # GAE advantage estimation
    advantages = np.zeros(T, dtype=np.float32)
    returns    = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_val = buf.values[t + 1] if (t + 1 < T and not buf.dones[t]) else 0.0
        delta    = buf.rewards[t] + GAMMA * next_val - buf.values[t]
        gae      = delta + GAMMA * GAE_LAMBDA * (1.0 - float(buf.dones[t])) * gae
        advantages[t] = gae
        returns[t]    = advantages[t] + buf.values[t]

    adv_t  = torch.tensor(advantages, dtype=torch.float32, device=device)
    adv_t  = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    b_s    = torch.stack(buf.spatials).to(device)
    b_p    = torch.stack(buf.proprios).to(device)
    b_m    = torch.stack(buf.masks).to(device)
    b_a    = torch.tensor(buf.actions, dtype=torch.int64, device=device)
    b_lp   = torch.tensor(buf.logprobs, dtype=torch.float32, device=device)
    b_ret  = torch.tensor(returns, dtype=torch.float32, device=device)
    b_h    = torch.cat(buf.hiddens, dim=0).to(device)

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


# ---------------------------------------------------------------------------
# Improvement chart
# ---------------------------------------------------------------------------
def save_improvement_chart(history: List[Dict], num_agents: int, out_path: str):
    """
    Saves a grouped bar chart:
      Green  = evacuated count
      Red    = dead count
      Gray   = trapped (neither evacuated nor dead by episode end)
    Plus a cyan line for evacuation rate %.
    """
    episodes  = [h["episode"]   for h in history]
    evacuated = [h["evacuated"] for h in history]
    dead      = [h["dead"]      for h in history]
    trapped   = [h["trapped"]   for h in history]
    evac_rate = [100.0 * h["evacuated"] / num_agents for h in history]

    fig, ax1 = plt.subplots(figsize=(max(10, len(episodes) * 0.8), 5))
    x = np.arange(len(episodes))
    w = 0.28

    bars_e = ax1.bar(x - w,     evacuated, w, color="#2ecc71", label="Evacuated",  zorder=3)
    bars_d = ax1.bar(x,         dead,      w, color="#e74c3c", label="Dead",        zorder=3)
    bars_t = ax1.bar(x + w,     trapped,   w, color="#95a5a6", label="Trapped",     zorder=3)

    ax1.set_xlabel("Episode", fontsize=12)
    ax1.set_ylabel("Agent Count", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Ep {e}" for e in episodes], rotation=45, ha='right', fontsize=9)
    ax1.set_ylim(0, num_agents + 1)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax1.set_axisbelow(True)

    # Evacuation rate line on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(x, evac_rate, color="#00bcd4", linewidth=2.5, marker='o', markersize=5,
             label="Evac Rate %", zorder=4)
    ax2.set_ylabel("Evacuation Rate (%)", fontsize=12, color="#00bcd4")
    ax2.tick_params(axis='y', colors="#00bcd4")
    ax2.set_ylim(0, 105)

    # Annotate bar values
    for bar in bars_e:
        h = bar.get_height()
        if h > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.1, str(int(h)),
                     ha='center', va='bottom', fontsize=7, color='#27ae60')

    # Combined legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2,
               loc='upper left', fontsize=9, framealpha=0.9)

    plt.title("RL Training Progress: Evacuation Improvement per Episode\n"
              "(Ground Floor | Corridor Chokepoint Fire | ActorCriticRNN with PPO)",
              fontsize=11, pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  → Chart saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    # --- Banner ---
    BAR = "━" * 68
    print(f"\n{BAR}")
    print(f"  🚀  RL EVACUATION TRAINER")
    print(f"{BAR}")
    print(f"  Floor    : Ground Floor (data/floors/o2_/g.png)")
    print(f"  Episodes : {args.episodes}")
    print(f"  Agents   : {args.agents}  (60% Adult / 30% Young / 10% Elderly)")
    print(f"  Max Ticks: {args.ticks}")
    print(f"  Curriculum: {'Enabled (spawn radius expands over episodes)' if not args.no_curriculum else 'Disabled'}")
    print(f"  Output   : {os.path.abspath(OUT_DIR)}")
    print(f"{BAR}\n")

    # --- Load floor ---
    print("  Loading ground floor...", end=" ", flush=True)
    img, free_space, graph, exits, _ = load_floor(FLOOR_KEY)
    lengths, _ = nx.multi_source_dijkstra(graph, sources=exits, weight='weight')
    dist_map    = {graph.nodes[n]['pos']: d for n, d in lengths.items()}
    exit_coords = [graph.nodes[e]['pos'] for e in exits]
    base_bgr    = cv2.cvtColor(np.asarray(img, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    print(f"done  ({len(exits)} exits, {len(dist_map):,} navigable positions)")

    if not exit_coords:
        print("  ❌ ERROR: No exit nodes found in floor plan XML. Cannot train.")
        sys.exit(1)

    # --- Build env template (we re-use same free_space / dist_map each episode) ---
    env = HazardEvacEnv(
        free_space_grid=free_space,
        dist_map=dist_map,
        exit_coords=exit_coords,
        num_agents=args.agents,
        age_ratio=(6, 3, 1),
        max_ticks=args.ticks,
        fire_origin=FIRE_ORIGIN,
        exit_reach_dist=25.0,
        exit_reward=300.0,
    )
    n_spawn = len(env.graph_positions)
    print(f"  Spawn pool: {n_spawn} corridor graph nodes")

    if n_spawn < args.agents:
        print(f"  ⚠️  Warning: spawn pool ({n_spawn}) < agents ({args.agents}). Reducing agents to {n_spawn}.")
        args.agents = n_spawn
        # Rebuild env with corrected num_agents
        env = HazardEvacEnv(
            free_space_grid=free_space, dist_map=dist_map, exit_coords=exit_coords,
            num_agents=args.agents, max_ticks=args.ticks, fire_origin=FIRE_ORIGIN,
            exit_reach_dist=25.0, exit_reward=300.0,
        )

    # --- Initialize policy (always from scratch) ---
    device = torch.device("cpu")
    policy = ActorCriticRNN().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    print(f"  Policy: ActorCriticRNN ({sum(p.numel() for p in policy.parameters()):,} params) — fresh random weights\n")

    # --- Training loop ---
    history: List[Dict] = []
    total_start = time.time()

    print(f"  {'Ep':>4}  {'Cur.Dist':>9}  {'Evac':>12}  {'Dead':>5}  {'Stuck':>6}  {'Reward':>8}  {'GradNorm':>9}  Files")
    print(f"  {'-'*4}  {'-'*9}  {'-'*12}  {'-'*5}  {'-'*6}  {'-'*8}  {'-'*9}  -----")

    for ep in range(1, args.episodes + 1):
        cur_dist = get_curriculum_dist(ep, args.episodes, not args.no_curriculum)
        cur_label = f"{int(cur_dist)}px" if cur_dist is not None else " full "

        buf = RolloutBuffer()
        result, s_png, e_png, v_path = run_episode(
            env, base_bgr, policy, buf, ep, args, cur_dist
        )

        gnorm = ppo_update(policy, optimizer, buf, device)

        history.append({**result, "gnorm": gnorm})

        evac = result["evacuated"]
        dead = result["dead"]
        stuck = result["trapped"]
        total = args.agents
        pct  = int(100 * evac / total)
        rwd  = result["mean_reward"]

        evac_str = f"{evac}/{total} ({pct:3d}%)"
        print(
            f"  {ep:4d}  {cur_label:>9}  {evac_str:>12}  {dead:5d}  {stuck:6d}  "
            f"{rwd:8.2f}  {gnorm:9.4f}  ep{ep:02d}_start.png | ep{ep:02d}_end.png | ep{ep:02d}.mp4"
        )

    elapsed = time.time() - total_start

    # --- Improvement chart ---
    chart_path = os.path.join(OUT_DIR, "improvement_chart.png")
    summary_path = os.path.join(OUT_DIR, "summary.txt")

    print(f"\n  Generating improvement chart...")
    save_improvement_chart(history, args.agents, chart_path)

    # --- Best episode ---
    best = max(history, key=lambda h: h["evacuated"])

    # --- Summary text ---
    lines = [
        "RL TRAINING SUMMARY",
        "=" * 50,
        f"Floor         : Ground Floor (g.png)",
        f"Episodes      : {args.episodes}",
        f"Agents        : {args.agents}",
        f"Max Ticks     : {args.ticks}",
        f"Fire Origin   : {FIRE_ORIGIN} (corridor chokepoint)",
        f"Curriculum    : {'Enabled' if not args.no_curriculum else 'Disabled'}",
        "",
        f"Episode Results:",
    ]
    for h in history:
        pct = int(100 * h["evacuated"] / args.agents)
        lines.append(
            f"  Ep {h['episode']:02d}: Evac={h['evacuated']:2d}/{args.agents} ({pct:3d}%)"
            f"  Dead={h['dead']:2d}  Stuck={h['trapped']:2d}  Reward={h['mean_reward']:7.2f}"
        )
    lines += [
        "",
        f"Best Episode  : Ep {best['episode']:02d} — {best['evacuated']}/{args.agents} evacuated",
        f"Total Time    : {elapsed:.0f}s ({elapsed/60:.1f} min)",
        "",
        "Output Files:",
        f"  Improvement chart : {chart_path}",
        f"  Per-episode PNGs  : {OUT_DIR}/ep??_start.png, ep??_end.png",
        f"  Per-episode videos: {OUT_DIR}/ep??.mp4",
    ]
    if args.save_checkpoint:
        lines.append(f"  Checkpoint saved  : {CKPT_FILE}")

    summary_text = "\n".join(lines)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    # --- Checkpoint ---
    if args.save_checkpoint:
        policy.save(CKPT_FILE)
        print(f"  💾 Weights saved → {CKPT_FILE}")

    # --- Final banner ---
    BAR2 = "━" * 68
    print(f"\n{BAR2}")
    print(f"  📊  RESULTS")
    print(f"{BAR2}")
    print(f"  Best episode : Ep {best['episode']:02d}  — {best['evacuated']}/{args.agents} evacuated")
    first_pct = int(100 * history[0]["evacuated"] / args.agents)
    last_pct  = int(100 * history[-1]["evacuated"] / args.agents)
    print(f"  Trend        : {first_pct}% → {last_pct}%  (episode 1 → {args.episodes})")
    print(f"  Improvement  : +{last_pct - first_pct} percentage points")
    print(f"  Time elapsed : {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    print(f"{BAR2}")
    print(f"\n  Open these files to see the results:")
    print(f"  • Chart  : {os.path.abspath(chart_path)}")
    print(f"  • Videos : {os.path.abspath(OUT_DIR)}\\ep01.mp4 ... ep{args.episodes:02d}.mp4")
    print(f"  • Summary: {os.path.abspath(summary_path)}")
    print()


if __name__ == "__main__":
    main()
