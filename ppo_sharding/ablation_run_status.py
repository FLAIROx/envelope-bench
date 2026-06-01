"""Audit the SAC ablation runs in wandb: which cells ran through, which crashed/failed, and why.

The project accumulated runs from three array attempts (two failed infra attempts +
the final array 17452330). This filters to the final array by creation time, dedups
retries per (variant, seed) cell, reports how far each got and its final return, and
maps crashed/failed cells back to their slurm error log via task_id = seed*6 + variant_idx.
"""

import argparse
import re
import subprocess
from collections import defaultdict

import wandb

RUN_RE = re.compile(r"ablation_(?P<tag>.+)_seed(?P<seed>\d+)$")
VARIANT_IDX = {"full": 0, "no_utd": 1, "no_obsnorm": 2, "no_criticnorm": 3, "no_3layers": 4, "baseline": 5}
TOTAL_TIMESTEPS = 1_500_000_000
DONE_STEP = int(TOTAL_TIMESTEPS * 0.99)  # treat >=99% as "ran through"

# Final-array runs were created at/after this UTC time; earlier clusters are dead attempts.
FINAL_ARRAY_AFTER = "2026-05-30T22:59:00"
SLURM_JOB = "17452330"
LOGDIR = "slurm_scripts/slurmlogs"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--entity", default="aneeshmuppidi19")
    p.add_argument("--project", default="envelope-sac-ablation")
    p.add_argument("--after", default=FINAL_ARRAY_AFTER, help="UTC ISO cutoff for the final array")
    p.add_argument("--job", default=SLURM_JOB)
    return p.parse_args()


def err_signature(task_id, job, logdir):
    """Return a one-line cause from the slurm .err for this task, if available."""
    path = f"{logdir}/slurm-panda_open_cabinet_sac_ablation-{job}_{task_id}.err"
    try:
        with open(path) as f:
            text = f.read()
    except FileNotFoundError:
        return f"(no log at {path})"
    for needle, label in [
        ("DUE TO TIME LIMIT", "slurm TIMEOUT (walltime)"),
        ("awaitable_signals_contract", "orbax checkpoint save hang (TimeoutError)"),
        ("SIGBUS", "SIGBUS (venv/mmap)"),
        ("mjx_panda.xml", "mujoco asset read error"),
        ("Stale file handle", "NFS stale file handle"),
    ]:
        if needle in text:
            return label
    # fall back to last Error/Exception line
    for line in reversed(text.splitlines()):
        if re.search(r"Error|Exception|error:", line):
            return line.strip()[:100]
    return "(no obvious error signature)"


def main():
    args = parse_args()
    api = wandb.Api()
    runs = list(api.runs(f"{args.entity}/{args.project}"))

    final, earlier = [], 0
    for r in runs:
        ca = str(getattr(r, "created_at", ""))[:19]
        if ca >= args.after:
            final.append(r)
        else:
            earlier += 1

    # cell (tag, seed) -> list of run dicts
    cells = defaultdict(list)
    for r in final:
        m = RUN_RE.search(str(r.config.get("run_name", "")))
        if not m:
            continue
        tag, seed = m.group("tag"), int(m.group("seed"))
        cells[(tag, seed)].append({
            "state": r.state,
            "step": r.summary.get("_step") or 0,
            "ret": r.summary.get("episode/return"),
            "created": str(getattr(r, "created_at", ""))[:19],
        })

    print("=" * 92)
    print(f"SAC ablation final-array audit  ({args.entity}/{args.project}, job {args.job})")
    print(f"{len(final)} runs in final array ({earlier} older runs from dead attempts ignored)")
    print("=" * 92)
    header = f"{'variant':<15}{'seed':>4}{'task':>5}  {'cell status':<26}{'reached':>14}{'final ret':>11}  cause-if-not-clean"
    print(header)
    print("-" * 92)

    ran_through, clean_finish, needs_note = 0, 0, []
    for tag in sorted(VARIANT_IDX, key=lambda t: VARIANT_IDX[t]):
        for seed in sorted({s for (t, s) in cells if t == tag}):
            attempts = cells[(tag, seed)]
            task_id = seed * 6 + VARIANT_IDX[tag]
            best = max(attempts, key=lambda a: a["step"])
            reached_end = best["step"] >= DONE_STEP
            any_finished = any(a["state"] == "finished" and a["step"] >= DONE_STEP for a in attempts)
            if reached_end:
                ran_through += 1
            if any_finished:
                status, cause = "clean finish", ""
                clean_finish += 1
            elif reached_end:
                status = "trained 1.5B, end-error"
                cause = err_signature(task_id, args.job, LOGDIR)
                needs_note.append((tag, seed, task_id))
            else:
                status = "did NOT finish"
                cause = err_signature(task_id, args.job, LOGDIR)
                needs_note.append((tag, seed, task_id))
            ret = best["ret"]
            rstr = f"{ret:11.1f}" if isinstance(ret, (int, float)) else f"{'-':>11}"
            natt = f" ({len(attempts)} attempts)" if len(attempts) > 1 else ""
            print(f"{tag:<15}{seed:>4}{task_id:>5}  {status:<26}{best['step']:>14,}{rstr}  {cause}{natt}")

    print("-" * 92)
    n_cells = len(cells)
    print(f"cells: {n_cells}/18 present | ran through (>=99% steps): {ran_through} | clean wandb finish: {clean_finish}")
    if needs_note:
        print("\nCells whose 'best' run is not a clean finish (data still valid if it reached 1.5B):")
        for tag, seed, tid in needs_note:
            print(f"  {tag}_seed{seed}  -> slurm {args.job}_{tid}.err")
    print("=" * 92)


if __name__ == "__main__":
    main()
