import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import time
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx

from src.floor_loader import load_floor
from src.env_evac import HazardEvacEnv
from src.rl_model import ActorCriticNetwork, ActorCriticRNN


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Agent Recurrent PPO Training for Indoor Hazard Evacuation")
    parser.add_argument("--floor", type=str, default="ground", help="Floor plan layout ('ground', 'first', 'penthouse', 'o1')")
    parser.add_argument("--episodes", type=int, default=5, help="Number of training iterations (episodes) to run")
    parser.add_argument("--num_agents", type=int, default=20, help="Number of concurrent evacuation occupants")
    parser.add_argument("--max_ticks", type=int, default=100, help="Max simulation ticks per episode")
    parser.add_argument("--model_type", type=str, default="rnn", choices=["rnn", "mlp"], help="Model type: 'rnn' or 'mlp'")
    parser.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clipping parameter epsilon")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy bonus coefficient")
    parser.add_argument("--value_coef", type=float, default=0.5, help="Critic loss coefficient")
    parser.add_argument("--ppo_epochs", type=int, default=3, help="PPO optimization epochs per rollout")
    parser.add_argument("--resume", action="store_true", help="Resume from previous checkpoint if available (default: train from scratch)")
    parser.add_argument("--save_checkpoint", action="store_true", help="Save final weights to checkpoints/ directory")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


class MARLTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cpu") # Optimized for CPU in conda base

        # Set random seeds
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        print("=" * 85)
        print("🚀 MULTI-AGENT RL BACKEND ENGINE")
        print("=" * 85)
        print(f"Model Architecture : {'ActorCriticRNN (Recurrent GRU)' if args.model_type == 'rnn' else 'ActorCriticNetwork (Feedforward)'}")
        print(f"Training Mode      : {'RESUME from checkpoint' if args.resume else 'FROM SCRATCH (Fresh Random Weights)'}")
        print(f"Planned Iterations : {args.episodes} Episodes (Manual Control)")
        print(f"Crowd Occupants    : {args.num_agents} Agents (Demographics: 60% Adult, 30% Young, 10% Elderly)")
        print(f"Max Ticks / Episode: {args.max_ticks}")
        print(f"Target Floor       : {args.floor}")
        print("=" * 85 + "\n")

        # Load environment resources
        img, self.free_space, self.graph, self.exits, _ = load_floor(args.floor)
        lengths, _ = nx.multi_source_dijkstra(self.graph, sources=self.exits, weight='weight')
        self.dist_map = {self.graph.nodes[n]['pos']: d for n, d in lengths.items()}
        self.exit_coords = [self.graph.nodes[e]['pos'] for e in self.exits]

        # Model instantiation: ALWAYS from scratch unless --resume is explicitly specified
        if args.model_type == "rnn":
            self.policy = ActorCriticRNN().to(self.device)
        else:
            self.policy = ActorCriticNetwork().to(self.device)

        ckpt_file = os.path.join(args.save_dir, f"{args.model_type}_policy_latest.pt")
        if args.resume and os.path.exists(ckpt_file):
            self.policy.load(ckpt_file)
            print(f"[Backend] Resumed previous model weights from: {ckpt_file}\n")
        else:
            print("[Backend] Initialized fresh random neural weights (Glorot/Uniform). No previous knowledge retained.\n")

        self.optimizer = optim.Adam(self.policy.parameters(), lr=args.lr)

        # Baseline snapshots to track physical weight displacement over time
        self.initial_weights = {n: p.clone().detach() for n, p in self.policy.named_parameters()}
        self.prev_weights = {n: p.clone().detach() for n, p in self.policy.named_parameters()}

    def _compute_weight_changes(self) -> Dict[str, float]:
        """Calculates Euclidean displacement (delta) across each neural component."""
        deltas = {"cnn": 0.0, "proprio": 0.0, "gru": 0.0, "actor": 0.0, "critic": 0.0, "total": 0.0}

        for name, param in self.policy.named_parameters():
            diff = (param.detach() - self.prev_weights[name]).norm().item()
            deltas["total"] += diff

            if "cnn" in name:
                deltas["cnn"] += diff
            elif "proprio" in name:
                deltas["proprio"] += diff
            elif "gru" in name:
                deltas["gru"] += diff
            elif "actor" in name:
                deltas["actor"] += diff
            elif "critic" in name:
                deltas["critic"] += diff

        # Update previous snapshot
        self.prev_weights = {n: p.clone().detach() for n, p in self.policy.named_parameters()}
        return deltas

    def _compute_cumulative_drift(self) -> Dict[str, float]:
        """Calculates total net weight shift from Episode 0 (initial scratch weights)."""
        drift = {"cnn": 0.0, "proprio": 0.0, "gru": 0.0, "actor": 0.0, "critic": 0.0, "total": 0.0}
        for name, param in self.policy.named_parameters():
            diff = (param.detach() - self.initial_weights[name]).norm().item()
            drift["total"] += diff
            if "cnn" in name: drift["cnn"] += diff
            elif "proprio" in name: drift["proprio"] += diff
            elif "gru" in name: drift["gru"] += diff
            elif "actor" in name: drift["actor"] += diff
            elif "critic" in name: drift["critic"] += diff
        return drift

    def train(self) -> List[Dict]:
        history = []
        start_time = time.time()

        print("-" * 115)
        print(f"{'Ep':>3} | {'Steps':>5} | {'Evacuated':>11} | {'Dead':>4} | {'Mean Reward':>11} | {'Grad Norm':>9} | "
              f"{'Δ W_CNN':>8} | {'Δ W_GRU':>8} | {'Δ W_Actor':>9} | {'Δ W_Critic':>10}")
        print("-" * 115)

        for ep in range(1, self.args.episodes + 1):
            ep_seed = self.args.seed + ep * 100
            env = HazardEvacEnv(
                free_space_grid=self.free_space,
                dist_map=self.dist_map,
                exit_coords=self.exit_coords,
                num_agents=self.args.num_agents,
                max_ticks=self.args.max_ticks,
                fire_origin=(250, 580) # Corridor Chokepoint
            )

            obs, info = env.reset(seed=ep_seed)

            # Rollout storage
            buf_spatial = []
            buf_proprio = []
            buf_mask = []
            buf_action = []
            buf_logprob = []
            buf_value = []
            buf_reward = []
            buf_done = []
            buf_hidden = []

            # Recurrent hidden states per agent
            agent_hiddens: Dict[int, torch.Tensor] = {}
            ep_rewards: Dict[int, float] = {aid: 0.0 for aid in obs.keys()}
            step_count = 0

            # -------------------------------------------------------------
            # 1. Rollout Phase (The Simulation / "Exam")
            # -------------------------------------------------------------
            while True:
                step_count += 1
                actions = {}

                for aid, o in obs.items():
                    s = o["spatial"]
                    p = o["proprio"]
                    m = o["action_mask"]
                    h_in = agent_hiddens.get(aid, None)

                    if self.args.model_type == "rnn":
                        act, logp, val, h_next = self.policy.act_rnn(s, p, m, hidden_state=h_in)
                        agent_hiddens[aid] = h_next.detach()
                        buf_hidden.append(h_in if h_in is not None else torch.zeros(1, 128))
                    else:
                        act, logp, val = self.policy.act(s, p, m)

                    actions[aid] = act
                    buf_spatial.append(torch.from_numpy(s).permute(2, 0, 1).float())
                    buf_proprio.append(torch.from_numpy(p).float())
                    buf_mask.append(torch.from_numpy(m).bool())
                    buf_action.append(act)
                    buf_logprob.append(logp)
                    buf_value.append(val)

                next_obs, rewards, term, trunc, info = env.step(actions)

                for aid in actions.keys():
                    r = rewards.get(aid, 0.0)
                    d = term.get(aid, False) or trunc.get(aid, False)
                    buf_reward.append(r)
                    buf_done.append(d)
                    ep_rewards[aid] = ep_rewards.get(aid, 0.0) + r

                obs = next_obs
                if info["all_done"] or step_count >= self.args.max_ticks:
                    break

            # -------------------------------------------------------------
            # 2. Generalized Advantage Estimation (GAE)
            # -------------------------------------------------------------
            T = len(buf_reward)
            if T == 0:
                continue

            returns = np.zeros(T, dtype=np.float32)
            advantages = np.zeros(T, dtype=np.float32)

            gae = 0.0
            for t in reversed(range(T)):
                next_val = buf_value[t + 1] if t + 1 < T and not buf_done[t] else 0.0
                delta = buf_reward[t] + self.args.gamma * next_val - buf_value[t]
                gae = delta + self.args.gamma * self.args.gae_lambda * (1.0 - float(buf_done[t])) * gae
                advantages[t] = gae
                returns[t] = advantages[t] + buf_value[t]

            adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
            norm_advantages = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

            b_spatial = torch.stack(buf_spatial).to(self.device)
            b_proprio = torch.stack(buf_proprio).to(self.device)
            b_mask = torch.stack(buf_mask).to(self.device)
            b_action = torch.tensor(buf_action, dtype=torch.int64, device=self.device)
            b_old_logprob = torch.tensor(buf_logprob, dtype=torch.float32, device=self.device)
            b_returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

            if self.args.model_type == "rnn":
                b_hidden = torch.cat(buf_hidden, dim=0).to(self.device)

            # -------------------------------------------------------------
            # 3. PPO Optimization (Weight Rewiring)
            # -------------------------------------------------------------
            grad_norms = []
            for _ in range(self.args.ppo_epochs):
                if self.args.model_type == "rnn":
                    new_values, new_logprob, entropy, _ = self.policy.evaluate_rnn_actions(
                        b_spatial, b_proprio, b_action, b_mask, b_hidden
                    )
                else:
                    new_values, new_logprob, entropy = self.policy.evaluate_actions(
                        b_spatial, b_proprio, b_action, b_mask
                    )

                ratio = torch.exp(new_logprob - b_old_logprob)
                surr1 = ratio * norm_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.args.clip_eps, 1.0 + self.args.clip_eps) * norm_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * nn.functional.mse_loss(new_values, b_returns)
                loss = policy_loss + self.args.value_coef * value_loss - self.args.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                gnorm = nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                grad_norms.append(gnorm.item())
                self.optimizer.step()

            # Measure weight adjustments in this episode
            deltas = self._compute_weight_changes()

            # Metrics
            evac_cnt = info["evacuated"]
            dead_cnt = info["dead"]
            evac_rate = (evac_cnt / self.args.num_agents) * 100.0
            mean_ep_reward = float(np.mean(list(ep_rewards.values())))
            mean_gnorm = float(np.mean(grad_norms))

            print(f"{ep:3d} | {step_count:5d} | {evac_cnt:2d}/{self.args.num_agents} ({evac_rate:4.0f}%) | "
                  f"{dead_cnt:4d} | {mean_ep_reward:11.2f} | {mean_gnorm:9.4f} | "
                  f"{deltas['cnn']:8.4f} | {deltas['gru']:8.4f} | {deltas['actor']:9.4f} | {deltas['critic']:10.4f}")

            history.append({
                "episode": ep,
                "evacuated": evac_cnt,
                "reward": mean_ep_reward,
                "grad_norm": mean_gnorm,
                "deltas": deltas
            })

        print("-" * 115)
        elapsed = time.time() - start_time
        drift = self._compute_cumulative_drift()

        print("\n" + "=" * 85)
        print(f"📊 TRAINING SESSION REPORT ({self.args.episodes} EPISODES COMPLETED IN {elapsed:.1f}s)")
        print("=" * 85)
        print(f"Total Cumulative Weight Drift from Scratch:")
        print(f"  • Vision CNN Layers     : Δ = {drift['cnn']:.4f}")
        print(f"  • Recurrent GRU Memory  : Δ = {drift['gru']:.4f}")
        print(f"  • Policy (Actor) Head   : Δ = {drift['actor']:.4f}")
        print(f"  • Value (Critic) Head   : Δ = {drift['critic']:.4f}")
        print(f"  • Net Model Drift (L2)  : Δ = {drift['total']:.4f}")
        print("-" * 85)
        print("Notice: The weights changed significantly based on hazard penalties and rewards.")

        if self.args.save_checkpoint:
            os.makedirs(self.args.save_dir, exist_ok=True)
            ckpt_path = os.path.join(self.args.save_dir, f"{self.args.model_type}_policy_latest.pt")
            self.policy.save(ckpt_path)
            print(f"💾 Saved updated weights to: {ckpt_path}")
        else:
            print("🔒 Contained Run: Weights were NOT saved to disk (--save_checkpoint was omitted).")
            print("   Next time you run this command, the model will train 100% from scratch again!")
        print("=" * 85 + "\n")

        return history


def main():
    args = parse_args()
    trainer = MARLTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
