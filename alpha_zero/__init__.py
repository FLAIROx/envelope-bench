# alpha_zero: Gumbel AlphaZero with MCTX for Envelope-Bench
#
# Uses mctx.gumbel_muzero_policy for MCTS-based policy improvement.
# Supports any discrete-action envelope environment across all adapters
# (Gymnax, Jumanji, Craftax, NaviX, etc.).
#
# Parallelism strategy (same as ppo_sharding):
#   - num_envs environments are SHARDED across GPUs
#   - Network params, optimizer state are REPLICATED on all devices
#   - Gradients are automatically all-reduced by XLA
#   - MCTS runs per-env within the sharded batch
