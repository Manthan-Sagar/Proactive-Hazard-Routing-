import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import tempfile
import numpy as np
import torch
import pytest

from src.rl_model import ActorCriticNetwork, ActorCriticRNN


def test_network_init():
    """Verify layer construction and default parameter counts."""
    model = ActorCriticNetwork(spatial_channels=6, proprio_dim=7, action_dim=9)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0
    print(f"\n[Test] Feedforward Model created with {total_params:,} parameters.")

    rnn_model = ActorCriticRNN(spatial_channels=6, proprio_dim=7, action_dim=9)
    rnn_params = sum(p.numel() for p in rnn_model.parameters())
    assert rnn_params > total_params
    print(f"[Test] Recurrent (RNN / GRU) Model created with {rnn_params:,} parameters.")


def test_single_agent_act():
    """Verify single-agent observation inference."""
    model = ActorCriticNetwork()
    spatial = np.random.uniform(0.0, 1.0, size=(21, 21, 6)).astype(np.float32)
    proprio = np.random.uniform(-1.0, 1.0, size=(7,)).astype(np.float32)
    mask = np.ones((9,), dtype=np.int8)

    action, log_prob, value = model.act(spatial, proprio, mask)
    assert isinstance(action, int)
    assert 0 <= action < 9
    assert isinstance(log_prob, float)
    assert isinstance(value, float)
    print(f"[Test] Single inference: action={action}, log_prob={log_prob:.4f}, value={value:.4f}")


def test_rnn_act_and_memory():
    """Verify recurrent model maintains and updates hidden state over steps."""
    rnn_model = ActorCriticRNN()
    spatial = np.random.uniform(0.0, 1.0, size=(21, 21, 6)).astype(np.float32)
    proprio = np.random.uniform(-1.0, 1.0, size=(7,)).astype(np.float32)
    mask = np.ones((9,), dtype=np.int8)

    hidden = None
    for step in range(3):
        action, log_prob, value, hidden = rnn_model.act_rnn(spatial, proprio, mask, hidden_state=hidden)
        assert isinstance(action, int)
        assert hidden is not None
        assert hidden.shape == (1, 128)
    print("[Test] Recurrent GRU hidden state updated across 3 steps successfully.")


def test_action_masking():
    """Verify that masked actions are never selected."""
    model = ActorCriticNetwork()
    spatial = np.zeros((21, 21, 6), dtype=np.float32)
    proprio = np.zeros((7,), dtype=np.float32)

    # Allow only action 3 (East) and action 0 (Stay)
    mask = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.int8)

    for _ in range(30):
        action, _, _ = model.act(spatial, proprio, mask, deterministic=False)
        assert action in (0, 3), f"Invalid action {action} was sampled with mask {mask}!"
    print("[Test] Action masking successfully blocks prohibited actions over 30 samples.")


def test_batched_forward_and_backward():
    """Verify batched evaluation and gradient backpropagation for PPO updates."""
    batch_size = 16
    model = ActorCriticRNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    spatial = torch.randn(batch_size, 6, 21, 21)
    proprio = torch.randn(batch_size, 7)
    actions = torch.randint(0, 9, (batch_size,))
    mask = torch.ones(batch_size, 9)

    values, log_probs, entropy, next_hidden = model.evaluate_rnn_actions(spatial, proprio, actions, mask)

    assert values.shape == (batch_size,)
    assert log_probs.shape == (batch_size,)
    assert entropy.shape == (batch_size,)
    assert next_hidden.shape == (batch_size, 128)

    # Combined loss
    loss = values.mean() - log_probs.mean() - 0.01 * entropy.mean()
    optimizer.zero_grad()
    loss.backward()

    has_grads = any(p.grad is not None and p.grad.norm() > 0 for p in model.parameters())
    assert has_grads, "Backpropagation failed to produce gradients."
    optimizer.step()
    print(f"[Test] Batched RNN forward/backward pass passed with batch_size={batch_size}.")


def test_save_and_load():
    """Verify model weights can be saved and restored with identical outputs."""
    model = ActorCriticRNN()
    spatial = np.random.uniform(0.0, 1.0, size=(21, 21, 6)).astype(np.float32)
    proprio = np.random.uniform(-1.0, 1.0, size=(7,)).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        model.save(tmp_path)
        new_model = ActorCriticRNN()
        new_model.load(tmp_path)

        a1, _, v1, _ = model.act_rnn(spatial, proprio, deterministic=True)
        a2, _, v2, _ = new_model.act_rnn(spatial, proprio, deterministic=True)

        assert a1 == a2
        assert abs(v1 - v2) < 1e-5
        print("[Test] Checkpoint serialization and deserialization verified.")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    test_network_init()
    test_single_agent_act()
    test_rnn_act_and_memory()
    test_action_masking()
    test_batched_forward_and_backward()
    test_save_and_load()
    print("\n✅ All Actor-Critic tests passed successfully!")
