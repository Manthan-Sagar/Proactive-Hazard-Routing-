import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class SpatialCNN(nn.Module):
    """Convolutional visual feature extractor for egocentric grid (21 x 21 x 6)."""
    def __init__(self, in_channels: int = 6, out_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 21x21 -> 10x10
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 5)), # (B, 64, 5, 5)
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, out_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ProprioMLP(nn.Module):
    """Multi-layer perceptron for agent proprioception vector (R^7)."""
    def __init__(self, in_dim: int = 7, out_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class ActorCriticNetwork(nn.Module):
    """
    Multi-Agent Actor-Critic Neural Network for Proactive Fire Evacuation.

    Observation Architecture:
      1. Spatial CNN: Ingests local egocentric grid (21 x 21 x 6)
      2. Proprioception MLP: Ingests agent status vector (R^7)
      3. Fusion Trunk: 128 (CNN) + 32 (Proprio) -> 128
      4. Actor Head: Outputs 9-DoF discrete action distribution with action masking.
      5. Critic Head: Outputs scalar state value V(s).
    """

    def __init__(self, spatial_channels: int = 6, proprio_dim: int = 7, action_dim: int = 9):
        super().__init__()
        self.spatial_channels = spatial_channels
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim

        self.cnn = SpatialCNN(spatial_channels, out_dim=128)
        self.proprio_mlp = ProprioMLP(proprio_dim, out_dim=32)

        self.fusion = nn.Sequential(
            nn.Linear(128 + 32, 128),
            nn.ReLU()
        )

        self.actor_head = nn.Linear(128, action_dim)
        self.critic_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def _prepare_inputs(
        self,
        spatial: Union[np.ndarray, torch.Tensor],
        proprio: Union[np.ndarray, torch.Tensor],
        action_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if device is None:
            device = next(self.parameters()).device

        if isinstance(spatial, np.ndarray):
            spatial = torch.from_numpy(spatial).float()
        if spatial.ndim == 3:
            spatial = spatial.permute(2, 0, 1).unsqueeze(0)
        elif spatial.ndim == 4 and spatial.shape[-1] == self.spatial_channels:
            spatial = spatial.permute(0, 3, 1, 2)

        if isinstance(proprio, np.ndarray):
            proprio = torch.from_numpy(proprio).float()
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)

        if action_mask is not None:
            if isinstance(action_mask, np.ndarray):
                action_mask = torch.from_numpy(action_mask).bool()
            if action_mask.ndim == 1:
                action_mask = action_mask.unsqueeze(0)
            action_mask = action_mask.to(device)

        return spatial.to(device), proprio.to(device), action_mask

    def forward(
        self,
        spatial: torch.Tensor,
        proprio: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        spatial_feat = self.cnn(spatial)
        proprio_feat = self.proprio_mlp(proprio)
        latent = self.fusion(torch.cat([spatial_feat, proprio_feat], dim=-1))

        logits = self.actor_head(latent)

        if action_mask is not None:
            mask_bool = action_mask.bool()
            all_masked = (~mask_bool).all(dim=-1, keepdim=True)
            safe_mask = mask_bool.clone()
            safe_mask[:, 0] = torch.where(all_masked.squeeze(-1), True, safe_mask[:, 0])

            neg_inf = torch.tensor(-1e8, device=logits.device, dtype=logits.dtype)
            logits = torch.where(safe_mask, logits, neg_inf)

        value = self.critic_head(latent).squeeze(-1)
        return logits, value

    def act(
        self,
        spatial: Union[np.ndarray, torch.Tensor],
        proprio: Union[np.ndarray, torch.Tensor],
        action_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        deterministic: bool = False
    ) -> Tuple[Union[int, np.ndarray], np.ndarray, np.ndarray]:
        device = next(self.parameters()).device
        s_tensor, p_tensor, m_tensor = self._prepare_inputs(spatial, proprio, action_mask, device)

        with torch.no_grad():
            logits, value = self.forward(s_tensor, p_tensor, m_tensor)
            dist = Categorical(logits=logits)

            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)

        act_res = action.cpu().numpy()
        logp_res = log_prob.cpu().numpy()
        val_res = value.cpu().numpy()

        if act_res.size == 1:
            return int(act_res.item()), float(logp_res.item()), float(val_res.item())
        return act_res, logp_res, val_res

    def evaluate_actions(
        self,
        spatial: torch.Tensor,
        proprio: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(spatial, proprio, action_mask)
        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy()

        return values, log_probs, dist_entropy

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str, map_location: str = "cpu") -> None:
        state_dict = torch.load(filepath, map_location=map_location)
        self.load_state_dict(state_dict)


class ActorCriticRNN(ActorCriticNetwork):
    """
    Recurrent (RNN / GRU) Actor-Critic Network for Partially Observable (Dec-POMDP) Evacuation.

    Maintains an internal hidden state vector (128 dimensions) over time, allowing agents
    to remember previously seen fire boundaries, dead-ends, and congestion patterns
    even when they step outside their local 21 x 21 vision window.
    """

    def __init__(self, spatial_channels: int = 6, proprio_dim: int = 7, action_dim: int = 9, rnn_hidden_dim: int = 128):
        super().__init__(spatial_channels, proprio_dim, action_dim)
        self.rnn_hidden_dim = rnn_hidden_dim

        # Recurrent GRU memory unit
        self.gru = nn.GRUCell(128, rnn_hidden_dim)

        # Actor & Critic heads operate on recurrent memory representation
        self.actor_head = nn.Linear(rnn_hidden_dim, action_dim)
        self.critic_head = nn.Sequential(
            nn.Linear(rnn_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward_rnn(
        self,
        spatial: torch.Tensor,
        proprio: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward step with recurrent hidden state transition.
        Returns:
            logits: Masked action logits (B, 9)
            value: Estimated state value V(s) (B,)
            next_hidden: Updated recurrent memory (B, 128)
        """
        spatial_feat = self.cnn(spatial)
        proprio_feat = self.proprio_mlp(proprio)
        fused = self.fusion(torch.cat([spatial_feat, proprio_feat], dim=-1))

        if hidden_state is None:
            hidden_state = torch.zeros(fused.size(0), self.rnn_hidden_dim, device=fused.device)

        next_hidden = self.gru(fused, hidden_state)

        logits = self.actor_head(next_hidden)

        if action_mask is not None:
            mask_bool = action_mask.bool()
            all_masked = (~mask_bool).all(dim=-1, keepdim=True)
            safe_mask = mask_bool.clone()
            safe_mask[:, 0] = torch.where(all_masked.squeeze(-1), True, safe_mask[:, 0])

            neg_inf = torch.tensor(-1e8, device=logits.device, dtype=logits.dtype)
            logits = torch.where(safe_mask, logits, neg_inf)

        value = self.critic_head(next_hidden).squeeze(-1)
        return logits, value, next_hidden

    def act_rnn(
        self,
        spatial: Union[np.ndarray, torch.Tensor],
        proprio: Union[np.ndarray, torch.Tensor],
        action_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        hidden_state: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[Union[int, np.ndarray], np.ndarray, np.ndarray, torch.Tensor]:
        """
        Single-step rollout action selection with recurrent memory update.
        """
        device = next(self.parameters()).device
        s_tensor, p_tensor, m_tensor = self._prepare_inputs(spatial, proprio, action_mask, device)

        if hidden_state is not None:
            hidden_state = hidden_state.to(device)

        with torch.no_grad():
            logits, value, next_hidden = self.forward_rnn(s_tensor, p_tensor, m_tensor, hidden_state)
            dist = Categorical(logits=logits)

            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)

        act_res = action.cpu().numpy()
        logp_res = log_prob.cpu().numpy()
        val_res = value.cpu().numpy()

        if act_res.size == 1:
            return int(act_res.item()), float(logp_res.item()), float(val_res.item()), next_hidden
        return act_res, logp_res, val_res, next_hidden

    def evaluate_rnn_actions(
        self,
        spatial: torch.Tensor,
        proprio: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluates batch transitions for recurrent PPO updates."""
        logits, values, next_hidden = self.forward_rnn(spatial, proprio, action_mask, hidden_state)
        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy()

        return values, log_probs, dist_entropy, next_hidden
