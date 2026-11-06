import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCriticNetwork(nn.Module):
    """
    Multi-Agent Actor-Critic Neural Network for Proactive Fire Evacuation.

    Observation Architecture:
      1. Spatial CNN: Ingests local egocentric grid (21 x 21 x 6)
         - Channels: [FreeSpace, Occupancy, Padding, Fire, HazardDistance, NormalizedPotential]
      2. Proprioception MLP: Ingests agent status vector (R^7)
         - Features: [DemographicOneHot (3), SpeedTicksFrac, DistToExit, DistToFire, DirToExitX]
      3. Fusion Trunk: Concatenates visual and proprioceptive representations.
      4. Actor Head: Outputs 9-DoF discrete action distribution with action masking.
      5. Critic Head: Outputs scalar state value V(s).
    """

    def __init__(self, spatial_channels: int = 6, proprio_dim: int = 7, action_dim: int = 9):
        super().__init__()
        self.spatial_channels = spatial_channels
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim

        # 1. Spatial CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(spatial_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 21x21 -> 10x10
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 5)), # (B, 64, 5, 5)
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU()
        )

        # 2. Proprioception MLP
        self.proprio_mlp = nn.Sequential(
            nn.Linear(proprio_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # 3. Fusion Backbone (128 CNN features + 32 Proprio features = 160)
        self.fusion = nn.Sequential(
            nn.Linear(128 + 32, 128),
            nn.ReLU()
        )

        # 4. Policy (Actor) Head
        self.actor_head = nn.Linear(128, action_dim)

        # 5. Value (Critic) Head
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
        """Converts inputs to tensors and ensures proper shapes and device."""
        if device is None:
            device = next(self.parameters()).device

        # Ensure spatial is (Batch, Channels, Height, Width)
        if isinstance(spatial, np.ndarray):
            spatial = torch.from_numpy(spatial).float()
        if spatial.ndim == 3: # (H, W, C) -> (1, C, H, W)
            spatial = spatial.permute(2, 0, 1).unsqueeze(0)
        elif spatial.ndim == 4 and spatial.shape[-1] == self.spatial_channels: # (B, H, W, C) -> (B, C, H, W)
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
        """
        Forward pass computing masked action logits and state value.
        """
        # Spatial visual features
        spatial_feat = self.cnn(spatial)

        # Proprioceptive features
        proprio_feat = self.proprio_mlp(proprio)

        # Fused state representation
        latent = self.fusion(torch.cat([spatial_feat, proprio_feat], dim=-1))

        # Raw action logits
        logits = self.actor_head(latent)

        # Action Masking: mask out collision/invalid directions with large negative logits
        if action_mask is not None:
            mask_bool = action_mask.bool()
            # If all actions are somehow masked, fallback to stay (action 0)
            all_masked = (~mask_bool).all(dim=-1, keepdim=True)
            safe_mask = mask_bool.clone()
            safe_mask[:, 0] = torch.where(all_masked.squeeze(-1), True, safe_mask[:, 0])

            neg_inf = torch.tensor(-1e8, device=logits.device, dtype=logits.dtype)
            logits = torch.where(safe_mask, logits, neg_inf)

        # State value estimation
        value = self.critic_head(latent).squeeze(-1)

        return logits, value

    def act(
        self,
        spatial: Union[np.ndarray, torch.Tensor],
        proprio: Union[np.ndarray, torch.Tensor],
        action_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        deterministic: bool = False
    ) -> Tuple[Union[int, np.ndarray], np.ndarray, np.ndarray]:
        """
        Inference method for agent stepping during rollout.
        Returns:
            action (int or np.ndarray): Sampled or greedy action index.
            log_prob (np.ndarray): Log probability of selected action.
            value (np.ndarray): Critic's estimated state value.
        """
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
        """
        Evaluates batch transitions for PPO surrogate loss and GAE value updates.
        Returns:
            values: Estimated state values V(s).
            log_probs: Log probabilities log pi(a | s).
            dist_entropy: Policy entropy for exploration regularization.
        """
        logits, values = self.forward(spatial, proprio, action_mask)
        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy()

        return values, log_probs, dist_entropy

    def save(self, filepath: str) -> None:
        """Saves network weights."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str, map_location: str = "cpu") -> None:
        """Loads network weights."""
        state_dict = torch.load(filepath, map_location=map_location)
        self.load_state_dict(state_dict)

