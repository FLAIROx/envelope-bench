import dataclasses
import datetime
import json
import os
import time

import h5py
import jax
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx

import wandb


class Logger:
    """Logger for single-run data-parallel training.

    Metrics are logged directly (no multi-run buffering needed since
    env sharding parallelizes within a single run, not across runs).

    Metrics are stored in HDF5 format with appendable datasets:
        - steps: (num_logged_steps,)
        - {metric}: (num_logged_steps,)

    Local logging structure:
        runs/{env_name}/{timestamp}/
            config.json
            metrics.h5
            checkpoints/step_{global_step}/
    """

    def __init__(self, args):
        self.args = args
        self.log_every = args.log_every
        self.start_time: float | None = None
        self._prev_flush_time: float | None = None
        self._prev_flush_step: int = 0
        self._sps: float = 0

        # Setup run directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_suffix = getattr(args, "run_name", None)
        leaf = f"{timestamp}_{run_suffix}" if run_suffix else timestamp
        self.run_name = f"{args.env_name}_{leaf}"
        self.run_dir = os.path.abspath(os.path.join(args.log_dir, args.env_name, leaf))
        try:
            os.makedirs(self.run_dir, exist_ok=True)
        # linux filesystem doesnt allow for :: in directory names
        except OSError as e:
            print(
                f"Error creating run directory {self.run_dir}: {e}, "
                f"saving to {self.run_dir.replace('::', '-')}."
            )
            self.run_dir = self.run_dir.replace("::", "-")
            os.makedirs(self.run_dir, exist_ok=True)

        # Save config
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(dataclasses.asdict(args), f, indent=2)

        # Setup HDF5 metrics file
        self._metrics_path = os.path.join(self.run_dir, "metrics.h5")
        self._h5file = h5py.File(self._metrics_path, "w")

        # Setup wandb
        self._wandb_run = None
        if args.use_wandb:
            self._wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config=dataclasses.asdict(args),
                name=self.run_name,
            )

        # Orbax checkpointer for saving model state
        self._checkpointer = ocp.StandardCheckpointer()

        print(f"Run directory: {self.run_dir}")

    def log(self, global_steps, metrics):
        """Called from jax.debug.callback inside jit. Logs metrics directly."""
        step = int(global_steps)
        metrics = jax.tree.map(float, metrics)

        if step - self._prev_flush_step < self.log_every:
            return

        # Update SPS from wall time between log calls
        now = time.time()
        if self._prev_flush_time is not None:
            dt = now - self._prev_flush_time
            if dt > 0:
                self._sps = (step - self._prev_flush_step) / dt
        self._prev_flush_time = now
        self._prev_flush_step = step

        # Compute elapsed time
        elapsed = now - self.start_time if self.start_time is not None else 0.0

        # Append to HDF5
        self._h5_append("steps", step)
        self._h5_append("sps", self._sps)
        self._h5_append("time/total_time", elapsed)
        for key, value in metrics.items():
            self._h5_append(key, value)
        self._h5file.flush()

        # Print
        print(
            f"step: {step}, "
            f"episode/return: {metrics['episode/return']:.2f}"
            f"±{metrics['episode/return_std']:.2f}, "
            f"sps: {self._sps:.0f}, "
            f"policy/loss: {metrics['policy/loss']:.4f}, "
            f"value/loss: {metrics['value/loss']:.4f}"
        )

        if self._wandb_run is not None:
            wandb_data = {**metrics, "time/sps": self._sps, "time/total_time": elapsed}
            wandb.log(wandb_data, step=step)

    def _h5_append(self, key: str, value):
        """Append a scalar to a resizable HDF5 dataset."""
        f = self._h5file
        if key not in f:
            f.create_dataset(key, shape=(0,), maxshape=(None,), dtype="f8")
        ds = f[key]
        ds.resize(ds.shape[0] + 1, axis=0)
        ds[-1] = value

    def save_checkpoint(self, global_step, train_state):
        """Save checkpoint of the single training run.

        Gathers any sharded arrays and moves the full state to CPU.
        """
        _, state = nnx.split(train_state)
        cpu = jax.devices("cpu")[0]
        # Gather sharded arrays (XLA collects from all devices) and move to CPU
        state = jax.tree.map(lambda x: jax.device_put(x, cpu), state)

        path = os.path.join(self.run_dir, "checkpoints", f"step_{global_step}")
        try:
            self._checkpointer.save(path, state)
        except ValueError:
            # mujoco_playground state has arrays with zero size. If so, we save without
            # the train_state without the current env_state
            env_state = train_state.env_state
            state.env_state = None
            self._checkpointer.save(path, state)
            state.env_state = env_state

        print(f"Started saving checkpoint (will finish asynchronously): {path}")

    def log_once(self, data: dict):
        """Log one-time data like compile/lower times."""
        for k, v in data.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
            self._h5file.attrs[k] = v
        self._h5file.flush()

        if self._wandb_run is not None:
            wandb.run.summary.update(data)

    def finish(self):
        self._checkpointer.wait_until_finished()
        self._h5file.close()
        if self._wandb_run is not None:
            wandb.finish()
