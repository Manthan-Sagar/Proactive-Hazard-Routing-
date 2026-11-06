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
    parser.add_argument("--episodes", type=int, default=5, help="Number of training episodes")
    parser.add_argument("--num_agents", type=int, default=20, help="Number of concurrent evacuation occupants")
    parser.add_argument("--max_ticks", type=int, default=120, help="Max simulation ticks per episode")
    parser.add_argument("--model_type", type=str, default="rnn", choices=["rnn", "mlp"], help="Model type: 'rnn' or 'mlp'")
    parser.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clipping parameter epsilon")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy bonus coefficient")
    parser.add_argument("--value_coef", type=float, default=0.5, help="Critic loss coefficient")
    parser.add_argument("--ppo_epochs", type=int, default=3, help="PPO optimization epochs per rollout")
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

        print("=" * 70)
        print("🚀 INITIALIZING MULTI-AGENT REINFORCEMENT LEARNING TRAINER")
        print("=" * 70)
        print(f"Model Architecture : {'ActorCriticRNN (Recurrent GRU)' if args.model_type == 'rnn' else 'ActorCriticNetwork (Feedforward)'}")
        print(f"Floor Layout       : {args.floor}")
        print(f"Agents             : {args.num_agents}")
        print(f"Episodes           : {args.episodes}")
        print(f"Max Ticks / Episode: {args.max_ticks}")
        print(f"Device             : {self.device}")
        print("=" * 70 + "\n")

        # Load environment resources
        img, self.free_space, self.graph, self.exits, _ = load_floor(args.floor)
        lengths, _ = nx.multi_source_dijkstra(self.graph, sources=self.exits, weight='weight')
        self.dist_map = {self.graph.nodes[n]['pos']: d for n, d in lengths.items()}
        self.exit_coords = [self.graph.nodes[e]['pos'] for e in self.exits]

        # Model instantiation
        if args.model_type == "rnn":
            self.policy = ActorCriticRNN().to(self.device)
        else:
            self.policy = ActorCriticNetwork().to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=args.lr)
        os.makedirs(args.save_dir, exist_ok=True)

    def train(self) -> List[Dict]:
        history = []
        start_time = time.time()

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
            # 1. Rollout Phase
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

                    buf_spatial.append(torch.from_numpy(s).permute(2, 0, 1).float()) # (6, 21, 21)
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

            # Normalize advantages
            adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
            adv_mean = adv_tensor.mean()
            adv_std = adv_tensor.std() + 1e-8
            norm_advantages = (adv_tensor - adv_mean) / adv_std

            # Prepare training tensors
            b_spatial = torch.stack(buf_spatial).to(self.device)
            b_proprio = torch.stack(buf_proprio).to(self.device)
            b_mask = torch.stack(buf_mask).to(self.device)
            b_action = torch.tensor(buf_action, dtype=torch.int64, device=self.device)
            b_old_logprob = torch.tensor(buf_logprob, dtype=torch.float32, device=self.device)
            b_returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

            if self.args.model_type == "rnn":
                b_hidden = torch.cat(buf_hidden, dim=0).to(self.device)

            # -------------------------------------------------------------
            # 3. PPO Optimization Epochs
            # -------------------------------------------------------------
            policy_losses, value_losses, entropies = [], [], []

            for _ in range(self.args.ppo_epochs):
                if self.args.model_type == "rnn":
                    new_values, new_logprob, entropy, _ = self.policy.evaluate_rnn_actions(
                        b_spatial, b_proprio, b_action, b_mask, b_hidden
                    )
                else:
                    new_values, new_logprob, entropy = self.policy.evaluate_actions(
                        b_spatial, b_proprio, b_action, b_mask
                    )

                # Ratio r(theta)
                ratio = torch.exp(new_logprob - b_old_logprob)

                # Clipped surrogate objective
                surr1 = ratio * norm_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.args.clip_eps, 1.0 + self.args.clip_eps) * norm_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function loss (MSE)
                value_loss = 0.5 * nn.functional.mse_loss(new_values, b_returns)

                # Total loss
                loss = policy_loss + self.args.value_coef * value_loss - self.args.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())

            # -------------------------------------------------------------
            # 4. Metrics & Logging
            # -------------------------------------------------------------
            evac_cnt = info["evacuated"]
            dead_cnt = info["dead"]
            evac_rate = (evac_cnt / self.args.num_agents) * 100.0
            mean_ep_reward = float(np.mean(list(ep_rewards.values())))

            mean_p_loss = float(np.mean(policy_losses))
            mean_v_loss = float(np.mean(value_losses))
            mean_entropy = float(np.mean(entropies))

            log_entry = {
                "episode": ep,
                "ticks": step_count,
                "transitions": T,
                "mean_reward": mean_ep_reward,
                "evacuated": evac_cnt,
                "dead": dead_cnt,
                "evac_rate": evac_rate,
                "policy_loss": mean_p_loss,
                "value_loss": mean_v_loss,
                "entropy": mean_entropy
            }
            history.append(log_entry)

            print(f"Episode {ep:02d}/{self.args.episodes:02d} | Ticks: {step_count:3d} | "
                  f"Evacuated: {evac_cnt:2d}/{self.args.num_agents} ({evac_rate:5.1f}%) | "
                  f"Dead: {dead_cnt:2d} | Reward: {mean_ep_reward:7.2f} | "
                  f"PolLoss: {mean_p_loss:+.4f} | ValLoss: {mean_v_loss:.4f} | Ent: {mean_entropy:.3f}")

        # Save latest checkpoint
        ckpt_path = os.path.join(self.args.save_dir, f"{self.args.model_type}_policy_latest.pt")
        self.policy.save(ckpt_path)
        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"✅ TRAINING COMPLETE in {elapsed:.1f}s | Saved weights: {ckpt_path}")
        print("=" * 70)

        return history


def main():
    args = parse_args()
    trainer = MARLTrainer(args)
    history = trainer.train()


if __name__ == "__main__":
    main()
