# ppo_sharding: Data-Parallel PPO with JAX sharding + jit
#
# Parallelism strategy:
#   - num_envs environments are SHARDED across GPUs (each device steps a subset)
#   - Network params, optimizer state are REPLICATED on all devices
#   - Gradients are automatically all-reduced by XLA
#   - Global shuffle across devices for minibatch construction
#
# This uses:
#   - jax.sharding.Mesh + NamedSharding to annotate how arrays
#     are partitioned across devices
#   - jax.jit to compile the full program — XLA figures out
#     communication and parallelism automatically
#
# Key advantages over pmap + vmap:
#   1. True data-parallel within a single training run
#   2. Automatic gradient all-reduce — no explicit pmean
#   3. Flexible device mapping — works with any number of devices
#   4. Better compiler optimization — XLA sees the full program
#   5. Future-proof — pmap is a legacy API
