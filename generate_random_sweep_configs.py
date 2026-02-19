"""Generate per-environment random sweep configs from the template.

Reads the environment list from sweep_configs/sweep_config_base.yaml and
stamps out one YAML file per environment into sweep_configs_random/envs/.

Usage:
    uv run generate_random_sweep_configs.py
"""

from pathlib import Path

import yaml

BASE_CONFIG = Path("sweep_configs/sweep_config_base.yaml")
TEMPLATE = Path("sweep_configs_random/template.yaml")
OUT_DIR = Path("sweep_configs_random/envs")


def env_to_slug(env_name: str) -> str:
    """Convert env name to a safe filename, e.g. gymnax::CartPole-v1 -> gymnax__CartPole-v1."""
    return env_name.replace("::", "__")


def main():
    with open(BASE_CONFIG) as f:
        base = yaml.safe_load(f)

    env_names = base["parameters"]["env_name"]["values"]

    template_text = TEMPLATE.read_text()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for env_name in env_names:
        config_text = template_text.replace("ENV_NAME_HERE", env_name)
        out_path = OUT_DIR / f"{env_to_slug(env_name)}.yaml"
        out_path.write_text(config_text)

    print(f"Generated {len(env_names)} configs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
