#!/usr/bin/env python3
#
# Scores a VIO_Dashboard live run (backend/data/runs/<run_id>.jsonl) against
# user-supplied ground-truth waypoints (taped/surveyed positions from a
# physical test, e.g. the corners of a measured rectangular loop, or a
# corridor's start/end marks).
#
# Deliberately simpler than the EuRoC harness's alignSVD(): a live run has
# no dense ground-truth trajectory, only a handful of known points, so this
# does direct point-to-point matching instead of a full rigid-body
# trajectory alignment.
#
# Waypoint file format (JSON or YAML -- auto-detected by extension):
#   waypoints:
#     - label: "corner_A"
#       x: 0.0
#       y: 0.0
#       t_hint_s: 0.0      # optional: seconds since the run's first pose
#       tag: null          # optional: e.g. "low_texture" for the relaxed bar
#     - label: "corner_B"
#       x: 4.0
#       y: 0.0
#       t_hint_s: 12.4
#
# If t_hint_s is omitted, the waypoint is matched to whichever pose sample
# in the ENTIRE run is closest in (x, y) -- fine for a start/end mark that
# only occurs once, but ambiguous for a corner visited on multiple laps.
# For multi-lap runs, give each pass its own waypoint entry with a
# distinguishing label and a t_hint_s so each is matched unambiguously.
#
# Usage:
#   ./score_live_run.py --jsonl run.jsonl --waypoints waypoints.yaml
#   ./score_live_run.py --jsonl run.jsonl --waypoints waypoints.json \
#       --frame both --strict-bar-cm 5 --relaxed-bar-cm 30

import argparse
import json
import math
import sys


def load_waypoints(path):
    with open(path) as f:
        text = f.read()

    if path.endswith(".yaml") or path.endswith(".yml"):
        try:
            import yaml
        except ImportError:
            print(
                "error: PyYAML not installed, and waypoints file is .yaml -- "
                "either `pip install pyyaml` or supply a .json file instead",
                file=sys.stderr,
            )
            sys.exit(1)
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    waypoints = data["waypoints"]
    for w in waypoints:
        w.setdefault("z", 0.0)
        w.setdefault("t_hint_s", None)
        w.setdefault("tag", None)
    return waypoints


def load_poses(jsonl_path, frame):
    poses = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") != "pose":
                continue
            if d.get("frame") != frame:
                continue
            poses.append((d["t_ns"], d["position"]))
    poses.sort(key=lambda p: p[0])
    return poses


def planar_dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def match_waypoint(waypoint, poses, t0_ns):
    gt = (waypoint["x"], waypoint["y"], waypoint["z"])

    if waypoint["t_hint_s"] is not None:
        target_t_ns = t0_ns + int(waypoint["t_hint_s"] * 1e9)
        best = min(poses, key=lambda p: abs(p[0] - target_t_ns))
        return best

    # No time hint: fall back to nearest-in-(x,y) over the whole run. Only
    # reliable for a waypoint visited once (see module docstring).
    best = min(poses, key=lambda p: planar_dist(gt, p[1]))
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", required=True, help="path to the run's JSONL file")
    ap.add_argument("--waypoints", required=True, help="path to a waypoints JSON/YAML file")
    ap.add_argument(
        "--frame",
        choices=["raw", "corrected", "both"],
        default="corrected",
        help="which pose stream to score (default: corrected -- raw is reported "
        "alongside for visibility with --frame both, but only corrected is "
        "gated against the pass/fail bar)",
    )
    ap.add_argument(
        "--strict-bar-cm",
        type=float,
        default=5.0,
        help="pass/fail threshold for untagged waypoints, cm (default: 5, "
        "matching the product spec's textured-surface indoor target)",
    )
    ap.add_argument(
        "--relaxed-bar-cm",
        type=float,
        default=30.0,
        help="pass/fail threshold for waypoints tagged low_texture/dark, cm "
        "(default: 30, matching the product spec's dark-tunnel allowance)",
    )
    ap.add_argument(
        "--align-first-waypoint",
        action="store_true",
        help="shift the whole trajectory (translation only, no rotation) so "
        "the first waypoint's matched sample lines up exactly with its "
        "ground-truth position -- makes later errors reflect accumulated "
        "drift rather than a starting-frame offset. Off by default: this "
        "can hide a real initial-alignment error if used carelessly.",
    )
    args = ap.parse_args()

    waypoints = load_waypoints(args.waypoints)
    frames_to_score = ["raw", "corrected"] if args.frame == "both" else [args.frame]

    exit_code = 0

    for frame in frames_to_score:
        poses = load_poses(args.jsonl, frame)
        if len(poses) < 2:
            print(f"[{frame}] no poses found for this frame -- skipping", file=sys.stderr)
            continue

        t0_ns = poses[0][0]
        offset = (0.0, 0.0, 0.0)

        if args.align_first_waypoint and waypoints:
            first_wp = waypoints[0]
            matched_t, matched_p = match_waypoint(first_wp, poses, t0_ns)
            offset = (
                first_wp["x"] - matched_p[0],
                first_wp["y"] - matched_p[1],
                first_wp["z"] - matched_p[2],
            )

        def aligned(p):
            return (p[0] + offset[0], p[1] + offset[1], p[2] + offset[2])

        print(f"\n=== frame={frame} ===")
        if args.align_first_waypoint:
            print(f"  (translation-only alignment applied: {offset})")

        print(f"{'label':<20}{'gt (x,y)':<20}{'matched (x,y)':<20}{'t_s':<10}{'error_cm':<10}{'bar':<8}")

        errors_cm = []
        any_fail = False

        for w in waypoints:
            matched_t, matched_p = match_waypoint(w, poses, t0_ns)
            matched_p = aligned(matched_p)
            gt = (w["x"], w["y"])
            err_cm = planar_dist(gt, matched_p) * 100.0
            errors_cm.append(err_cm)

            bar = args.relaxed_bar_cm if w.get("tag") == "low_texture" else args.strict_bar_cm
            status = "PASS" if err_cm <= bar else "FAIL"
            if status == "FAIL":
                any_fail = True

            t_s = (matched_t - t0_ns) / 1e9
            print(
                f"{w['label']:<20}"
                f"({gt[0]:+.3f},{gt[1]:+.3f}){'':<4}"
                f"({matched_p[0]:+.3f},{matched_p[1]:+.3f}){'':<4}"
                f"{t_s:<10.2f}"
                f"{err_cm:<10.1f}"
                f"{bar:.0f}cm ({status})"
            )

        first_p = aligned(poses[0][1])
        last_p = aligned(poses[-1][1])
        endpoint_error_cm = planar_dist(first_p, last_p) * 100.0
        endpoint_status = "PASS" if endpoint_error_cm <= args.strict_bar_cm else "FAIL"
        if endpoint_status == "FAIL":
            any_fail = True

        max_err = max(errors_cm) if errors_cm else 0.0
        mean_err = sum(errors_cm) / len(errors_cm) if errors_cm else 0.0

        overall = "FAIL" if any_fail else "PASS"
        if frame == args.frame or args.frame == "both":
            if frame == "corrected" or args.frame != "both":
                if any_fail:
                    exit_code = 1

        print(
            f"\n[LIVE-EVAL SUMMARY] frame={frame} "
            f"max_waypoint_error_cm={max_err:.1f} "
            f"mean_waypoint_error_cm={mean_err:.1f} "
            f"endpoint_error_cm={endpoint_error_cm:.1f} "
            f"{overall} ({args.strict_bar_cm:.0f}cm bar)"
        )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
