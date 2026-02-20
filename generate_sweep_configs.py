"""Generate sweep config YAML files for PPO hyperparameter sweeps.

Modify BASELINE_DEFAULTS and SWEEPS below to change hyperparameter values,
then run: python generate_sweep_configs.py
"""

import os
from pathlib import Path

OUTPUT_DIR = Path("sweep_configs")

# ─── Baseline hyperparameter defaults ───────────────────────────────────────
# Every sweep config uses these values unless overridden by its sweep definition.
BASELINE_DEFAULTS = {
    "total_timesteps": "100663296",  # 2^25 *3
    "seed": 0,
    "num_envs": 1024,
    "num_steps": 64,
    "num_epochs": 4,
    "num_minibatches": 8,
    "policy_lr": "3e-4",
    "value_fn_lr": "1e-4",
    "policy_wd": 0.0001,
    "value_wd": 0.0001,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "entropy_coef": 0.01,
    "epsilon": 0.2,
    "normalize_observations": True,
    "normalize_rewards": False,
    "use_symlog": False,
    "layer_size": 256,
    "activation": "tanh",
    "stagger_steps": 1,
    "pool_size": 16,
    "num_runs": 8,
    "num_checkpoints": 1,
    "optimizer": "adamw",
    "anneal_lr": False,
    "adam_epsilon": "1e-5",
}

# ─── Wandb settings ─────────────────────────────────────────────────────────
WANDB_ENTITY = "flair"
WANDB_PROJECT = "big-ppo-sweep"

# ─── Sweep definitions ──────────────────────────────────────────────────────
# Each entry: (file_suffix, name_suffix, param_name, sweep_values)
# - sweep_values: a list means `values: [...]` (grid over alternatives)
#                 a single value means `value: X` (override the default)
# - param_name: None for the base config (no parameter is swept)
SWEEPS = [
    ("base",        "base",        None,                     None),
    ("act",         "activation",  "activation",             ["swish", "relu"]),
    ("adam_eps",    "adam-eps",    "adam_epsilon",             ["1e-6", "1e-7", "1e-8"]),
    ("anneal_lr",   "anneal_lr",   "anneal_lr",              True),
    ("critic_wd",   "critic_wd",   "value_wd",               ["0.00001", "0.001"]),
    ("ent",         "ent",         "entropy_coef",            0),
    ("epsilon",     "epsilon",     "epsilon",                 [0.1, 0.5]),
    ("gae_lambda",  "gae_lambda",  "gae_lambda",              [0.8, 0.5, 0.3]),
    ("gamma",       "gamma",       "gamma",                   [0.9, 0.999]),
    ("layer_sizes", "layer_sizes", "layer_size",              [32, 64, 128]),
    ("lr",          "lr",          "policy_lr",               ["5e-5", "1e-3"]),
    ("norm_obs",    "norm_obs",    "normalize_observations",  False),
    ("norm_rew",    "norm_rew",    "normalize_rewards",       True),
    ("num_envs",    "num_envs",    "num_envs",                [512, 2048]),
    ("num_epochs",  "num_epochs",  "num_epochs",              [2, 8]),
    ("num_steps",   "num_steps",   "num_steps",               [32, 128]),
    ("optimizer",   "optimizer",   "optimizer",               "muon"),
    ("policy_wd",   "policy_wd",   "policy_wd",               ["0.00001", "0.001"]),
    ("symlog",      "symlog",      "use_symlog",              True),
    ("value_lr",    "value_lr",    "value_fn_lr",             ["5e-5", "1e-3"]),
]

# ─── Environment list ────────────────────────────────────────────────────────
# Active envs are strings; commented-out envs are prefixed with "#"
ENV_NAMES = [
    "gymnax::Acrobot-v1",
    "gymnax::Asterix-MinAtar",
    "gymnax::BernoulliBandit-misc",
    "gymnax::Breakout-MinAtar",
    "gymnax::CartPole-v1",
    "gymnax::Catch-bsuite",
    "gymnax::DeepSea-bsuite",
    "gymnax::DiscountingChain-bsuite",
    "gymnax::FourRooms-misc",
    "gymnax::Freeway-MinAtar",
    "gymnax::GaussianBandit-misc",
    "gymnax::MNISTBandit-bsuite",
    "gymnax::MemoryChain-bsuite",
    "gymnax::MetaMaze-misc",
    "gymnax::MountainCar-v0",
    "gymnax::MountainCarContinuous-v0",
    "gymnax::Pendulum-v1",
    "gymnax::PointRobot-misc",
    "gymnax::Pong-misc",
    "gymnax::Reacher-misc",
    "# gymnax::SimpleBandit-bsuite",
    "gymnax::SpaceInvaders-MinAtar",
    "gymnax::Swimmer-misc",
    "gymnax::UmbrellaChain-bsuite",
    "brax::ant",
    "brax::halfcheetah",
    "brax::hopper",
    "brax::humanoid",
    "brax::humanoidstandup",
    "brax::inverted_double_pendulum",
    "brax::inverted_pendulum",
    "brax::pusher",
    "brax::reacher",
    "brax::walker2d",
    "# jumanji::BinPack-v2",
    "jumanji::CVRP-v1",
    "# jumanji::Cleaner-v0",
    "# jumanji::Connector-v2",
    "jumanji::FlatPack-v0",
    "jumanji::Game2048-v1",
    "jumanji::GraphColoring-v1",
    "# jumanji::JobShop-v0",
    "jumanji::Knapsack-v1",
    "# jumanji::LevelBasedForaging-v0",
    "jumanji::MMST-v0",
    "jumanji::Maze-v0",
    "jumanji::Minesweeper-v0",
    "jumanji::MultiCVRP-v0",
    "jumanji::PacMan-v1",
    "jumanji::RobotWarehouse-v0",
    "jumanji::RubiksCube-partly-scrambled-v0",
    "jumanji::RubiksCube-v0",
    "jumanji::SlidingTilePuzzle-v0",
    "jumanji::Snake-v1",
    "jumanji::Sokoban-v0",
    "jumanji::Sudoku-v0",
    "jumanji::Sudoku-very-easy-v0",
    "jumanji::TSP-v1",
    "jumanji::Tetris-v0",
    "craftax::Craftax-Symbolic-v1",
    "craftax::Craftax-Classic-Symbolic-v1",
    "navix::Navix-DistShift1-v0",
    "navix::Navix-DistShift2-v0",
    "navix::Navix-DoorKey-16x16-v0",
    "navix::Navix-DoorKey-5x5-v0",
    "navix::Navix-DoorKey-6x6-v0",
    "navix::Navix-DoorKey-8x8-v0",
    "navix::Navix-DoorKey-Random-16x16-v0",
    "navix::Navix-DoorKey-Random-5x5-v0",
    "navix::Navix-DoorKey-Random-6x6-v0",
    "navix::Navix-DoorKey-Random-8x8-v0",
    "navix::Navix-Dynamic-Obstacles-16x16-v0",
    "navix::Navix-Dynamic-Obstacles-5x5-Random-v0",
    "navix::Navix-Dynamic-Obstacles-5x5-v0",
    "navix::Navix-Dynamic-Obstacles-6x6-Random-v0",
    "navix::Navix-Dynamic-Obstacles-6x6-v0",
    "navix::Navix-Dynamic-Obstacles-8x8-v0",
    "navix::Navix-Empty-16x16-v0",
    "navix::Navix-Empty-5x5-v0",
    "navix::Navix-Empty-6x6-v0",
    "navix::Navix-Empty-8x8-v0",
    "navix::Navix-Empty-Random-16x16-v0",
    "navix::Navix-Empty-Random-5x5-v0",
    "navix::Navix-Empty-Random-6x6-v0",
    "navix::Navix-Empty-Random-8x8-v0",
    "navix::Navix-FourRooms-v0",
    "navix::Navix-GoToDoor-5x5-v0",
    "navix::Navix-GoToDoor-6x6-v0",
    "navix::Navix-GoToDoor-8x8-v0",
    "navix::Navix-KeyCorridorS3R1-v0",
    "navix::Navix-KeyCorridorS3R2-v0",
    "navix::Navix-KeyCorridorS3R3-v0",
    "navix::Navix-KeyCorridorS4R3-v0",
    "navix::Navix-KeyCorridorS5R3-v0",
    "navix::Navix-KeyCorridorS6R3-v0",
    "navix::Navix-LavaGapS5-v0",
    "navix::Navix-LavaGapS6-v0",
    "navix::Navix-LavaGapS7-v0",
    "navix::Navix-SimpleCrossingS11N5-v0",
    "navix::Navix-SimpleCrossingS9N1-v0",
    "navix::Navix-SimpleCrossingS9N2-v0",
    "navix::Navix-SimpleCrossingS9N3-v0",
    "mujoco_playground::AcrobotSwingup",
    "mujoco_playground::AcrobotSwingupSparse",
    "mujoco_playground::AeroCubeRotateZAxis",
    "mujoco_playground::AlohaHandOver",
    "mujoco_playground::AlohaSinglePegInsertion",
    "mujoco_playground::ApolloJoystickFlatTerrain",
    "mujoco_playground::BallInCup",
    "mujoco_playground::BarkourJoystick",
    "mujoco_playground::BerkeleyHumanoidJoystickFlatTerrain",
    "mujoco_playground::BerkeleyHumanoidJoystickRoughTerrain",
    "mujoco_playground::CartpoleBalance",
    "mujoco_playground::CartpoleBalanceSparse",
    "mujoco_playground::CartpoleSwingup",
    "mujoco_playground::CartpoleSwingupSparse",
    "mujoco_playground::CheetahRun",
    "mujoco_playground::FingerSpin",
    "mujoco_playground::FingerTurnEasy",
    "mujoco_playground::FingerTurnHard",
    "mujoco_playground::FishSwim",
    "mujoco_playground::G1JoystickFlatTerrain",
    "mujoco_playground::G1JoystickRoughTerrain",
    "mujoco_playground::Go1Footstand",
    "mujoco_playground::Go1Getup",
    "mujoco_playground::Go1Handstand",
    "mujoco_playground::Go1JoystickFlatTerrain",
    "mujoco_playground::Go1JoystickRoughTerrain",
    "mujoco_playground::H1InplaceGaitTracking",
    "mujoco_playground::H1JoystickGaitTracking",
    "mujoco_playground::HopperHop",
    "mujoco_playground::HopperStand",
    "mujoco_playground::HumanoidRun",
    "mujoco_playground::HumanoidStand",
    "mujoco_playground::HumanoidWalk",
    "mujoco_playground::LeapCubeReorient",
    "mujoco_playground::LeapCubeRotateZAxis",
    "mujoco_playground::Op3Joystick",
    "mujoco_playground::PandaOpenCabinet",
    "mujoco_playground::PandaPickCube",
    "mujoco_playground::PandaPickCubeCartesian",
    "mujoco_playground::PandaPickCubeOrientation",
    "mujoco_playground::PandaRobotiqPushCube",
    "mujoco_playground::PendulumSwingup",
    "mujoco_playground::PointMass",
    "mujoco_playground::ReacherEasy",
    "mujoco_playground::ReacherHard",
    "mujoco_playground::SpotFlatTerrainJoystick",
    "mujoco_playground::SpotGetup",
    "mujoco_playground::SpotJoystickGaitTracking",
    "mujoco_playground::SwimmerSwimmer6",
    "mujoco_playground::T1JoystickFlatTerrain",
    "mujoco_playground::T1JoystickRoughTerrain",
    "mujoco_playground::WalkerRun",
    "mujoco_playground::WalkerStand",
    "mujoco_playground::WalkerWalk",
    "kinetix::s",
    "kinetix::m",
    "kinetix::l",
]


def fmt_value(v):
    """Format a single value for YAML output."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, str):
        return v
    if isinstance(v, float):
        # Avoid trailing zeros: 0.0001 -> "0.0001", 0.00001 -> "1e-05"
        # Match the original formatting
        s = f"{v:g}"
        return s
    return str(v)


def fmt_values_list(vals):
    """Format a list for YAML inline: [val1, val2, ...]"""
    return "[" + ", ".join(fmt_value(v) for v in vals) + "]"


def build_env_list_yaml():
    """Build the env_name values block."""
    lines = []
    for env in ENV_NAMES:
        if env.startswith("#"):
            lines.append(f"      {env.rstrip().replace('# ', '# - ')}")
        else:
            lines.append(f"      - {env}")
    return "\n".join(lines)


# The ordered list of parameters as they appear in the YAML files.
# adam_epsilon is handled separately since it only appears in base and adam_eps configs.
PARAM_ORDER = [
    "seed",
    "num_envs",
    "num_steps",
    "num_epochs",
    "num_minibatches",
    "policy_lr",
    "value_fn_lr",
    "policy_wd",
    "value_wd",
    "gamma",
    "gae_lambda",
    "entropy_coef",
    "epsilon",
    "normalize_observations",
    "normalize_rewards",
    "use_symlog",
    "layer_size",
    "activation",
    "stagger_steps",
    "pool_size",
    "num_runs",
    "num_checkpoints",
    "optimizer",
    "anneal_lr",
]


def generate_config(file_suffix, name_suffix, sweep_param, sweep_values):
    """Generate a single sweep config YAML string."""
    is_base = sweep_param is None
    is_adam_eps = sweep_param == "adam_epsilon"
    include_adam_epsilon = is_base or is_adam_eps

    # Determine if sweep_values is a list (grid search over alternatives)
    # or a single value (override default)
    if sweep_values is not None and isinstance(sweep_values, list):
        is_multi = True
    else:
        is_multi = False

    name = f"big-ppo-sweep-{name_suffix}"

    lines = []
    lines.append("command:")
    lines.append("  - python ")
    lines.append("  - -m")
    lines.append("  - ppo_vmap.ppo")
    lines.append("  - ${args}")
    lines.append(f"name: {name}")
    # Some names have a trailing space in the originals; we add one for
    # multi-word suffixes that had it, but it's cosmetic. Keep it consistent.
    lines.append("method: grid")
    lines.append("metric:")
    lines.append("  name: episode/return")
    lines.append("  goal: maximize")
    lines.append("")
    lines.append("parameters:")
    lines.append("  # --- Fixed Parameters \u2014")
    lines.append("  total_timesteps:")
    lines.append(f"    value: {BASELINE_DEFAULTS['total_timesteps']} # 2^25 *3")
    lines.append("  wandb_entity:")
    lines.append(f"    value: {WANDB_ENTITY}")
    lines.append("  wandb_project:")
    lines.append(f"    value: {WANDB_PROJECT}")
    lines.append("  use_wandb:")
    lines.append("    value: True")
    lines.append("")
    lines.append("  # --- Experiment Configuration ---")
    lines.append("  env_name:")
    lines.append("    values:")
    lines.append(build_env_list_yaml())

    # Write each parameter
    for param in PARAM_ORDER:
        default = BASELINE_DEFAULTS[param]

        key = f"  {param}:"

        if param == sweep_param:
            # This is the swept parameter
            lines.append(key)
            if is_multi:
                lines.append(f"    values: {fmt_values_list(sweep_values)}")
            else:
                lines.append(f"    value: {fmt_value(sweep_values)}")
        else:
            lines.append(key)
            lines.append(f"    value: {fmt_value(default)}")

    # adam_epsilon: only in base (as value) and adam_eps (as values)
    if include_adam_epsilon:
        lines.append("  adam_epsilon:")
        if is_adam_eps:
            lines.append(f"    values: {fmt_values_list(sweep_values)}")
        else:
            lines.append(f"    value: {fmt_value(BASELINE_DEFAULTS['adam_epsilon'])}")

    return "\n".join(lines) + "\n"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    for file_suffix, name_suffix, sweep_param, sweep_values in SWEEPS:
        filename = f"sweep_config_{file_suffix}.yaml"
        content = generate_config(file_suffix, name_suffix, sweep_param, sweep_values)
        path = OUTPUT_DIR / filename
        path.write_text(content)
        print(f"Generated {path}")

    print(f"\nDone! Generated {len(SWEEPS)} configs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
