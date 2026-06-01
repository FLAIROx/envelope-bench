"""Summarize the SAC leave-one-out ablation from wandb.

Pulls every `ablation_<variant>_seed<seed>` run from the wandb project, aggregates
the final `episode/return` across seeds per variant, computes each change's marginal
contribution (full -> no_X drop), prints a table, and logs the summary back to wandb
as a table + summary scalars (no plots).
"""

import argparse
import re
from collections import defaultdict

import numpy as np

import wandb

RUN_RE = re.compile(r"ablation_(?P<tag>.+)_seed(?P<seed>\d+)$")

# Display order and human labels for the leave-one-out variants.
VARIANT_ORDER = ["full", "no_utd", "no_obsnorm", "no_criticnorm", "no_3layers", "baseline"]
VARIANT_LABEL = {
    "full": "full (all changes)",
    "no_utd": "- UTD fix (8192 envs / batch 256)",
    "no_obsnorm": "- observation norm",
    "no_criticnorm": "- critic layer norm",
    "no_3layers": "- 3 layers (use 2)",
    "baseline": "baseline (all reverted)",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--entity", default="aneeshmuppidi19")
    p.add_argument("--project", default="envelope-sac-ablation")
    p.add_argument("--metric", default="episode/return")
    p.add_argument("--step_key", default="_step")
    p.add_argument(
        "--final_window",
        type=int,
        default=10,
        help="Average the last N logged points of the metric as the run's final score.",
    )
    p.add_argument("--summary_run_name", default="ablation_summary")
    return p.parse_args()


def run_final_score(run, metric, step_key, window, samples=500):
    """Return (final_mean_of_last_window, best, n_points) for a run, or None if no data.

    Uses wandb's server-side downsampled history (pandas=False -> list of dicts, numpy-only)
    instead of scan_history, which would stream every logged point (~75k/run for full runs)
    and is far too slow for runs that reached 1.5B steps.
    """
    rows = run.history(keys=[step_key, metric], samples=samples, pandas=False)
    pts = [(r[step_key], r[metric]) for r in rows if r.get(metric) is not None]
    if not pts:
        return None
    pts.sort(key=lambda t: t[0])
    vals = np.array([v for _, v in pts], dtype=np.float64)
    final = float(np.mean(vals[-window:])) if vals.size else float("nan")
    return final, float(np.max(vals)), int(vals.size)


def main():
    args = parse_args()
    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}")

    # (tag, seed) -> chosen run dict. Dedup across re-runs (and across the kept dead
    # attempts): prefer the run that reached the furthest step, breaking ties toward a
    # clean `finished` state. Max-step is robust even though downsampling caps n_points.
    best_by_key = {}
    skipped = []
    for run in runs:
        name = run.config.get("run_name") or run.name or ""
        m = RUN_RE.search(str(name))
        if not m:
            continue
        tag, seed = m.group("tag"), int(m.group("seed"))
        score = run_final_score(run, args.metric, args.step_key, args.final_window)
        if score is None:
            skipped.append(f"{tag}/seed{seed} ({run.state})")
            continue
        final, best, n = score
        max_step = run.summary.get(args.step_key) or 0
        finished = run.state == "finished"
        cand = {"final": final, "best": best, "n": n, "max_step": max_step,
                "finished": finished, "state": run.state}
        key = (tag, seed)
        cur = best_by_key.get(key)
        # prefer further-along run; tie-break to finished; then to higher max_step
        if cur is None or (max_step, finished) > (cur["max_step"], cur["finished"]):
            best_by_key[key] = cand

    # variant -> list of (seed, final, best, n_points)
    per_variant = defaultdict(list)
    for (tag, seed), c in best_by_key.items():
        per_variant[tag].append((seed, c["final"], c["best"], c["n"]))

    if not per_variant:
        raise SystemExit(
            f"No ablation runs with metric '{args.metric}' found in "
            f"{args.entity}/{args.project}. Are the runs finished and logged?"
        )

    # Aggregate across seeds.
    agg = {}  # tag -> dict
    for tag, entries in per_variant.items():
        finals = np.array([e[1] for e in entries], dtype=np.float64)
        bests = np.array([e[2] for e in entries], dtype=np.float64)
        agg[tag] = {
            "n_seeds": len(entries),
            "seeds": sorted(e[0] for e in entries),
            "final_mean": float(np.mean(finals)),
            "final_std": float(np.std(finals)),
            "best_mean": float(np.mean(bests)),
        }

    full_mean = agg.get("full", {}).get("final_mean")
    base_mean = agg.get("baseline", {}).get("final_mean")

    def marginal(tag):
        # How much removing this single change costs, at the tuned operating point.
        if full_mean is None or tag not in agg or tag in ("full", "baseline"):
            return None
        return full_mean - agg[tag]["final_mean"]

    # ---- Print summary table ----
    print("\n" + "=" * 78)
    print(f"SAC leave-one-out ablation  ({args.entity}/{args.project})")
    print(f"metric = {args.metric}  |  final = mean of last {args.final_window} logged points")
    print("=" * 78)
    header = f"{'variant':<34}{'seeds':>6}{'final (mean±std)':>22}{'Δ vs full':>14}"
    print(header)
    print("-" * 78)
    ordered = [t for t in VARIANT_ORDER if t in agg] + [t for t in agg if t not in VARIANT_ORDER]
    for tag in ordered:
        a = agg[tag]
        mc = marginal(tag)
        mc_str = "" if mc is None else f"{-mc:+.1f}"  # negative = drop from removing change
        label = VARIANT_LABEL.get(tag, tag)
        print(
            f"{label:<34}{a['n_seeds']:>6}"
            f"{a['final_mean']:>13.1f} ± {a['final_std']:<6.1f}{mc_str:>14}"
        )
    print("-" * 78)

    if full_mean is not None and base_mean is not None:
        total = full_mean - base_mean
        print(f"Total improvement (full - baseline): {total:+.1f}")
        contribs = {t: marginal(t) for t in ordered if marginal(t) is not None}
        if contribs:
            ranked = sorted(contribs.items(), key=lambda kv: kv[1], reverse=True)
            print("\nMarginal contribution ranking (most impactful change first):")
            for tag, mc in ranked:
                share = (mc / total * 100) if total else float("nan")
                print(f"  {VARIANT_LABEL.get(tag, tag):<34} {mc:+8.1f}  ({share:5.1f}% of total)")
            summed = sum(contribs.values())
            print(
                f"\n  Σ leave-one-out drops = {summed:+.1f} vs total {total:+.1f} "
                f"(gap = interaction/non-additivity = {total - summed:+.1f})"
            )
    if skipped:
        print(f"\nSkipped (no metric data yet): {', '.join(skipped)}")
    print("=" * 78 + "\n")

    # ---- Log summary back to wandb ----
    summary_run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.summary_run_name,
        job_type="analysis",
        config={"metric": args.metric, "final_window": args.final_window},
    )
    table = wandb.Table(
        columns=["variant", "label", "n_seeds", "final_mean", "final_std", "best_mean", "delta_vs_full"]
    )
    for tag in ordered:
        a = agg[tag]
        mc = marginal(tag)
        table.add_data(
            tag,
            VARIANT_LABEL.get(tag, tag),
            a["n_seeds"],
            round(a["final_mean"], 2),
            round(a["final_std"], 2),
            round(a["best_mean"], 2),
            None if mc is None else round(-mc, 2),
        )
    wandb.log({"ablation/summary": table})

    summary_scalars = {f"final_mean/{tag}": agg[tag]["final_mean"] for tag in ordered}
    summary_scalars.update({f"final_std/{tag}": agg[tag]["final_std"] for tag in ordered})
    for tag in ordered:
        mc = marginal(tag)
        if mc is not None:
            summary_scalars[f"marginal_drop/{tag}"] = mc
    if full_mean is not None and base_mean is not None:
        summary_scalars["total_improvement"] = full_mean - base_mean
    summary_run.summary.update(summary_scalars)
    wandb.finish()
    print(f"Logged summary to wandb run '{args.summary_run_name}' in {args.entity}/{args.project}")


if __name__ == "__main__":
    main()
