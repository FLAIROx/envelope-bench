import h5py
import json
import os
import sys

BASE = "runs/navix::Navix-Empty-5x5-v0"

# Collect all run data
runs = []
all_config_keys = set()

for d in sorted(os.listdir(BASE)):
    run_dir = os.path.join(BASE, d)
    config_path = os.path.join(run_dir, "config.json")
    metrics_path = os.path.join(run_dir, "metrics.h5")

    if not os.path.isdir(run_dir):
        continue

    # Read config
    with open(config_path, "r") as f:
        config = json.load(f)
    all_config_keys.update(config.keys())

    # Read metrics
    final_return = None
    steps = None
    mean_returns = None
    status = "NO_METRICS"
    try:
        with h5py.File(metrics_path, "r") as f:
            keys = list(f.keys())
            if "mean_return" in keys:
                mean_returns = f["mean_return"][:]
                steps = f["steps"][:]
                final_return = float(mean_returns[-1])
                status = "FINISHED"
            elif len(keys) == 0:
                status = "EMPTY"
            else:
                status = f"NO_MEAN_RETURN (keys: {keys})"
    except Exception as e:
        status = f"ERROR: {e}"

    runs.append({
        "dir": d,
        "config": config,
        "final_return": final_return,
        "steps": steps,
        "mean_returns": mean_returns,
        "status": status,
    })

# Sort config keys for consistent display
all_config_keys = sorted(all_config_keys)

# Exclude keys that are the same across all runs or not interesting for comparison
constant_keys = []
varying_keys = []
for k in all_config_keys:
    values = set()
    for r in runs:
        v = r["config"].get(k)
        values.add(str(v))
    if len(values) == 1:
        constant_keys.append(k)
    else:
        varying_keys.append(k)

print("=" * 120)
print("NAVIX-EMPTY-5x5-v0 SWEEP ANALYSIS")
print("=" * 120)

print(f"\nTotal runs found: {len(runs)}")
finished = [r for r in runs if r["status"] == "FINISHED"]
print(f"Finished runs (with metrics): {len(finished)}")
print(f"Empty/errored runs: {len(runs) - len(finished)}")

# Print constant config values
print("\n" + "-" * 80)
print("CONSTANT CONFIG VALUES (same across all runs):")
print("-" * 80)
for k in constant_keys:
    print(f"  {k:30s} = {runs[0]['config'].get(k)}")

# Print varying config values
print("\n" + "-" * 80)
print("VARYING CONFIG VALUES:")
print("-" * 80)
for k in varying_keys:
    values = sorted(set(str(r["config"].get(k)) for r in runs))
    print(f"  {k:30s} : {values}")

# Print full table
print("\n" + "=" * 120)
print("FULL RUN TABLE (sorted by final_return descending):")
print("=" * 120)

# Build header
header_keys = varying_keys  # Only show varying keys in the table for readability
header = f"{'Run Dir':20s} | {'Status':10s} | {'Final Return':>14s}"
for k in header_keys:
    header += f" | {k:>22s}"
print(header)
print("-" * len(header))

# Sort: finished runs by final_return desc, then unfinished
sorted_runs = sorted(runs, key=lambda r: (r["final_return"] is not None, r["final_return"] or -999), reverse=True)

for r in sorted_runs:
    fr_str = f"{r['final_return']:.4f}" if r['final_return'] is not None else "N/A"
    row = f"{r['dir']:20s} | {r['status']:10s} | {fr_str:>14s}"
    for k in header_keys:
        val = r["config"].get(k, "N/A")
        row += f" | {str(val):>22s}"
    print(row)

# Print the best and worst training curves
if len(finished) >= 2:
    best_run = max(finished, key=lambda r: r["final_return"])
    worst_run = min(finished, key=lambda r: r["final_return"])

    for label, run in [("BEST", best_run), ("WORST", worst_run)]:
        print(f"\n{'=' * 80}")
        print(f"{label} RUN: {run['dir']} (final_return={run['final_return']:.4f})")
        # Show its varying config
        for k in varying_keys:
            print(f"  {k}: {run['config'].get(k)}")
        print("-" * 80)
        print(f"{'Step':>10s} | {'Mean Return':>14s}")
        print("-" * 30)
        steps = run["steps"]
        returns = run["mean_returns"]
        # Print every 50th step for brevity, plus first and last
        n = len(steps)
        indices = list(range(0, n, 50))
        if (n - 1) not in indices:
            indices.append(n - 1)
        for i in indices:
            print(f"{int(steps[i]):>10d} | {returns[i]:>14.4f}")

elif len(finished) == 1:
    run = finished[0]
    print(f"\nOnly one finished run: {run['dir']} (final_return={run['final_return']:.4f})")

# Summary statistics
if finished:
    returns = [r["final_return"] for r in finished]
    print(f"\n{'=' * 80}")
    print("SUMMARY STATISTICS (across finished runs):")
    print(f"  Max final_return:  {max(returns):.4f}")
    print(f"  Min final_return:  {min(returns):.4f}")
    print(f"  Mean final_return: {sum(returns)/len(returns):.4f}")

# Optimal return discussion
print(f"\n{'=' * 80}")
print("OPTIMAL RETURN FOR Navix-Empty-5x5-v0:")
print("-" * 80)
print("Navix-Empty-5x5-v0 is a MiniGrid-style navigation task on a 5x5 grid.")
print("The agent starts at a random position and must navigate to a goal.")
print("The optimal path in a 5x5 grid is at most ~6 steps (Manhattan distance).")
print("Reward is typically 1 - 0.9*(step_count/max_steps). With max_steps often")
print("set to 100 for 5x5, and optimal ~4-6 steps, the max return per episode")
print("is approximately 1 - 0.9*(5/100) = 0.955, or similar depending on the")
print("exact reward shaping. In Navix, the reward is sparse: +1 on reaching goal,")
print("0 otherwise, with discount. So optimal return depends on discount and")
print("episode length. With gamma=0.99, the undiscounted optimal is close to 1.0")
print("if the agent reaches the goal quickly every episode.")
print()

# Check actual max values seen in training
if finished:
    overall_max = max(float(r["mean_returns"].max()) for r in finished)
    print(f"Highest mean_return seen during ANY training step: {overall_max:.4f}")
    # Also check per-run peaks
    for r in sorted(finished, key=lambda r: float(r["mean_returns"].max()), reverse=True)[:3]:
        peak = float(r["mean_returns"].max())
        peak_idx = r["mean_returns"].argmax()
        peak_step = int(r["steps"][peak_idx])
        print(f"  {r['dir']}: peak={peak:.4f} at step {peak_step}")

