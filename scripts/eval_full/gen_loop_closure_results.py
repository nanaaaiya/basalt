#!/usr/bin/env python3
#
# Parses the per-sequence log.txt files produced by run_loop_closure_eval.sh
# for the "[LOOP-CLOSURE SUMMARY] ..." line vio.cpp prints, and emits a
# results table (text + JSON) so accuracy changes can be diffed against a
# committed baseline instead of relying on someone remembering a number from
# a previous manual run.
#
# Usage: ./gen_loop_closure_results.py <out_dir> [--json baseline.json]

import argparse
import json
import os
import re
import sys

SUMMARY_RE = re.compile(
    r"\[LOOP-CLOSURE SUMMARY\] raw_start_to_end=([-\d.]+)m "
    r"corrected_start_to_end=([-\d.]+)m raw_ate_rmse=([-\d.]+)m "
    r"corrected_ate_rmse=([-\d.]+)m num_loop_closures=(\d+) "
    r"num_stored_keyframes=(\d+)"
)

FIELDS = [
    "raw_start_to_end",
    "corrected_start_to_end",
    "raw_ate_rmse",
    "corrected_ate_rmse",
    "num_loop_closures",
    "num_stored_keyframes",
]


def parse_log(path):
    with open(path) as f:
        for line in f:
            m = SUMMARY_RE.search(line)
            if m:
                vals = m.groups()
                return {
                    "raw_start_to_end": float(vals[0]),
                    "corrected_start_to_end": float(vals[1]),
                    "raw_ate_rmse": float(vals[2]),
                    "corrected_ate_rmse": float(vals[3]),
                    "num_loop_closures": int(vals[4]),
                    "num_stored_keyframes": int(vals[5]),
                }
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", help="directory produced by run_loop_closure_eval.sh")
    ap.add_argument("--json", help="also write a JSON summary to this path")
    ap.add_argument(
        "--baseline",
        help="a previously-saved JSON summary to diff corrected_ate_rmse against",
    )
    args = ap.parse_args()

    results = {}
    for d in sorted(os.listdir(args.out_dir)):
        log_path = os.path.join(args.out_dir, d, "log.txt")
        if not os.path.isfile(log_path):
            continue
        parsed = parse_log(log_path)
        if parsed is None:
            print(f"[warn] no [LOOP-CLOSURE SUMMARY] line found for {d}", file=sys.stderr)
            continue
        results[d] = parsed

    if not results:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    baseline = None
    if args.baseline and os.path.isfile(args.baseline):
        with open(args.baseline) as f:
            baseline = json.load(f)

    header = ["sequence"] + FIELDS
    if baseline:
        header.append("corrected_ate_rmse_delta")
    print(",".join(header))
    for d, r in results.items():
        row = [d] + [str(r[k]) for k in FIELDS]
        if baseline:
            if d in baseline:
                delta = r["corrected_ate_rmse"] - baseline[d]["corrected_ate_rmse"]
                row.append(f"{delta:+.4f}")
            else:
                row.append("n/a (no baseline)")
        print(",".join(row))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.json}", file=sys.stderr)

    if baseline:
        # Regression gate: corrected_ate_rmse must not regress by more than
        # 10% or 1cm (whichever is larger) vs. the committed baseline, and a
        # sequence that previously had loop closures must not drop to zero.
        regressed = []
        for d, r in results.items():
            if d not in baseline:
                continue
            b = baseline[d]
            tol = max(0.10 * b["corrected_ate_rmse"], 0.01)
            if r["corrected_ate_rmse"] > b["corrected_ate_rmse"] + tol:
                regressed.append(
                    f"{d}: corrected_ate_rmse {b['corrected_ate_rmse']:.4f}m -> "
                    f"{r['corrected_ate_rmse']:.4f}m (tol {tol:.4f}m)"
                )
            if b["num_loop_closures"] > 0 and r["num_loop_closures"] == 0:
                regressed.append(f"{d}: num_loop_closures dropped to 0 (was {b['num_loop_closures']})")

        print("")
        if regressed:
            print("REGRESSIONS DETECTED:")
            for line in regressed:
                print(f"  - {line}")
            sys.exit(2)
        else:
            print("No regressions vs. baseline.")


if __name__ == "__main__":
    main()
