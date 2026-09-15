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
# IMPORTANT -- frame alignment: VIO's world frame has an arbitrary yaw at
# startup (only gravity/roll/pitch is constrained by the init step, not
# heading), so tape-measured (x, y) coordinates will essentially never line
# up with VIO's own frame as-is. Use --align rigid (solves rotation +
# translation from your first two waypoints) for any waypoint file whose
# coordinates come from a real physical layout -- without it, a perfectly
# accurate trajectory can show a large false error purely from this
# rotation mismatch, not real VIO drift.
#
# Usage:
#   ./score_live_run.py --jsonl run.jsonl --waypoints waypoints.yaml --align rigid
#   ./score_live_run.py --jsonl run.jsonl --waypoints waypoints.json \
#       --frame both --align rigid --strict-bar-cm 5 --relaxed-bar-cm 30

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
        "--align",
        choices=["none", "translation", "rigid"],
        default="none",
        help="none (default): compare raw (x,y) as-is -- only valid if your "
        "ground-truth waypoints are already expressed in VIO's own world "
        "frame. translation: shift the trajectory so the FIRST waypoint "
        "lines up exactly, rotation untouched. rigid: translation AND "
        "rotation (2D, solved from the first TWO waypoints) -- needed "
        "whenever your waypoint (x,y) values come from a taped/measured "
        "layout, since VIO's world-frame yaw is arbitrary at startup (only "
        "gravity/roll/pitch is constrained by init, not heading) and will "
        "essentially never match your tape measure's coordinate axes. "
        "Without 'rigid', a perfectly accurate VIO trajectory can show a "
        "large false error purely from this frame-rotation mismatch.",
    )
    args = ap.parse_args()

    waypoints = load_waypoints(args.waypoints)
    frames_to_score = ["raw", "corrected"] if args.frame == "both" else [args.frame]

    # t_hint_s needs ONE shared "t=0" reference across both frames -- always
    # the raw frame's first pose, never each frame's own first pose. raw
    # publishes from the very first processed frame, but corrected doesn't
    # start until OnlineLoopClosure has produced its first valid smoothed
    # pose (needs at least one keyframe), which is measurably LATER. Using
    # each frame's own first sample as "t=0" silently points "t_hint_s=X"
    # at two different real moments for raw vs corrected -- this looks
    # exactly like a bad correction (wildly different matched positions)
    # but is actually just two mismatched clocks.
    raw_poses_for_t0 = load_poses(args.jsonl, "raw")
    if not raw_poses_for_t0:
        print("error: no raw poses found in this file -- can't establish a t=0 reference", file=sys.stderr)
        sys.exit(1)
    shared_t0_ns = raw_poses_for_t0[0][0]

    exit_code = 0

    for frame in frames_to_score:
        poses = load_poses(args.jsonl, frame)
        if len(poses) < 2:
            print(f"[{frame}] no poses found for this frame -- skipping", file=sys.stderr)
            continue

        # cos_a/sin_a implement a 2D rotation about (x,y); z only ever gets
        # a plain offset (roll/pitch, and therefore z, are gravity-
        # observable at init -- only yaw is arbitrary, so only x/y need a
        # rotation, never z).
        cos_a, sin_a = 1.0, 0.0
        offset = (0.0, 0.0, 0.0)

        if args.align in ("translation", "rigid") and waypoints:
            first_wp = waypoints[0]
            _, p1 = match_waypoint(first_wp, poses, shared_t0_ns)

            if args.align == "rigid":
                if len(waypoints) < 2:
                    print(
                        "error: --align rigid needs at least 2 waypoints to "
                        "solve for rotation",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                second_wp = waypoints[1]
                _, p2 = match_waypoint(second_wp, poses, shared_t0_ns)

                gt_dx = second_wp["x"] - first_wp["x"]
                gt_dy = second_wp["y"] - first_wp["y"]
                est_dx = p2[0] - p1[0]
                est_dy = p2[1] - p1[1]

                gt_angle = math.atan2(gt_dy, gt_dx)
                est_angle = math.atan2(est_dy, est_dx)
                rot = gt_angle - est_angle
                cos_a, sin_a = math.cos(rot), math.sin(rot)

            # Rotate p1 first (if rigid), then solve the translation that
            # makes the (possibly-rotated) first waypoint land exactly on
            # its ground-truth position.
            p1_rot_x = cos_a * p1[0] - sin_a * p1[1]
            p1_rot_y = sin_a * p1[0] + cos_a * p1[1]
            offset = (
                first_wp["x"] - p1_rot_x,
                first_wp["y"] - p1_rot_y,
                first_wp["z"] - p1[2],
            )

        def aligned(p):
            x = cos_a * p[0] - sin_a * p[1] + offset[0]
            y = sin_a * p[0] + cos_a * p[1] + offset[1]
            z = p[2] + offset[2]
            return (x, y, z)

        print(f"\n=== frame={frame} ===")
        if args.align != "none":
            rot_deg = math.degrees(math.atan2(sin_a, cos_a))
            print(f"  (align={args.align}: rotation={rot_deg:+.1f}deg, translation={offset})")

        print(f"{'label':<20}{'gt (x,y)':<20}{'matched (x,y)':<20}{'t_s':<10}{'error_cm':<10}{'bar':<8}")

        errors_cm = []
        any_fail = False

        for w in waypoints:
            matched_t, matched_p = match_waypoint(w, poses, shared_t0_ns)
            matched_p = aligned(matched_p)
            gt = (w["x"], w["y"])
            err_cm = planar_dist(gt, matched_p) * 100.0
            errors_cm.append(err_cm)

            bar = args.relaxed_bar_cm if w.get("tag") == "low_texture" else args.strict_bar_cm
            status = "PASS" if err_cm <= bar else "FAIL"
            if status == "FAIL":
                any_fail = True

            t_s = (matched_t - shared_t0_ns) / 1e9
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
