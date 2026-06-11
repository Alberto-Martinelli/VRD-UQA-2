"""Aggregate per-run summary.csv files into one comparison table.

Each parallel evaluation job writes a private run dir with its own summary.csv
(see run_vqa_analysis.sh). This read-only step concatenates them so models /
datasets evaluated in separate jobs can be compared in one table.

Run:
  uv run python VQA_analysis/metrics/4_aggregate_summaries.py [--tag NAME]
      [--run-glob 'eval_*'] [--datasets BDocs DUDE] [--run-ids r1 r2 ...]

Writes EVAL_RUNS_DIR/comparison_<tag>.csv. Honors VQA_EVAL_RUNS_DIR.
"""
import argparse
import csv
from config.run_layout import EVAL_RUNS_DIR, SUMMARY_COLUMNS

OUT_COLUMNS = ["run_id"] + SUMMARY_COLUMNS


def aggregate(run_glob="eval_*", tag="all", datasets=None, run_ids=None):
    if run_ids:
        run_dirs = [EVAL_RUNS_DIR / r for r in run_ids]
    else:
        run_dirs = sorted(p for p in EVAL_RUNS_DIR.glob(run_glob) if p.is_dir())

    rows = []
    used = []
    for rd in run_dirs:
        sc = rd / "summary.csv"
        if not sc.exists():
            continue
        used.append(rd.name)
        with open(sc, newline="") as f:
            for row in csv.DictReader(f):
                if datasets and row.get("dataset") not in datasets:
                    continue
                row["run_id"] = rd.name
                rows.append(row)

    out = EVAL_RUNS_DIR / f"comparison_{tag}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in OUT_COLUMNS})
    print(f"Wrote {out} ({len(rows)} rows from {len(used)} runs: {used})")
    return out


def main():
    p = argparse.ArgumentParser(description="Aggregate per-run summary.csv into a comparison table.")
    p.add_argument("--tag", default="all")
    p.add_argument("--run-glob", default="eval_*")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--run-ids", nargs="*", default=None)
    args = p.parse_args()
    aggregate(run_glob=args.run_glob, tag=args.tag, datasets=args.datasets, run_ids=args.run_ids)


if __name__ == "__main__":
    main()
