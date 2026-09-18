#!/usr/bin/env python3
#
# Scores a VIO_Dashboard live run (backend/data/runs/<run_id>.jsonl) captured
# with the CAMERA PHYSICALLY STATIONARY, to measure how badly a foreground
# disturbance (a hand or object waved close to the lens) corrupts the pose
# estimate -- and whether it recovers.
#
# Unlike score_live_run.py (which scores a walking loop against taped
# waypoints), there is no real trajectory here: ground truth is simply "the
# pose should not move at all". So instead of matching waypoints, this
# script:
#   1. Establishes a resting-position baseline (median position -- robust to
#      the brief excursions, since quiet time should dominate the run).
#   2. Flags contiguous stretches where the pose deviates from that baseline
#      beyond --excursion-thresh-cm as a "disturbance event".
#   3. Reports each event's peak excursion and how long it took to settle
#      back under the threshold.
#   4. Reports a residual offset: does the pose return to the ORIGINAL
#      resting position by the end of the run, or does it land somewhere
#      permanently shifted (a sign that corrupted landmarks got baked into
#      the map, which is worse than a transient wobble)?
#
# Run the same protocol (quiet / wave / quiet, a few repeats) once per
# calibration you're comparing, then run this script on each resulting
# .jsonl and compare the summary lines directly.
#
# Usage:
#   ./score_stationary_hold.py --jsonl run.jsonl
#   ./score_stationary_hold.py --jsonl run.jsonl --frame both \
#       --excursion-thresh-cm 5 --baseline-window-s 5

import argparse
import json
import math
import sys


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
            poses.append((d["t_ns"], tuple(d["position"])))
    poses.sort(key=lambda p: p[0])
    return poses


def dist3(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def median_position(positions):
    if not positions:
        return (0.0, 0.0, 0.0)
    xs = sorted(p[0] for p in positions)
    ys = sorted(p[1] for p in positions)
    zs = sorted(p[2] for p in positions)
    n = len(positions)
    mid = n // 2
    med = lambda s: s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0
    return (med(xs), med(ys), med(zs))


def linear_fit_slope(ts, values):
    """Ordinary least-squares slope of values against ts (seconds)."""
    n = len(ts)
    if n < 2:
        return 0.0
    mean_t = sum(ts) / n
    mean_v = sum(values) / n
    num = sum((t - mean_t) * (v - mean_v) for t, v in zip(ts, values))
    den = sum((t - mean_t) ** 2 for t in ts)
    return num / den if den > 0 else 0.0


def stddev(values):
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def find_events(poses, baseline, thresh_m, min_gap_s):
    """Contiguous stretches where distance-from-baseline exceeds thresh_m,
    merging stretches separated by a gap shorter than min_gap_s (so a single
    wave that briefly dips under threshold for a sample or two isn't split
    into several fake events)."""
    events = []
    cur_start_idx = None
    last_over_idx = None

    for i, (t_ns, pos) in enumerate(poses):
        over = dist3(pos, baseline) > thresh_m
        if over:
            if cur_start_idx is None:
                cur_start_idx = i
            last_over_idx = i
        else:
            if cur_start_idx is not None:
                gap_s = (t_ns - poses[last_over_idx][0]) / 1e9
                if gap_s > min_gap_s:
                    events.append((cur_start_idx, last_over_idx))
                    cur_start_idx = None

    if cur_start_idx is not None:
        events.append((cur_start_idx, last_over_idx))

    return events


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", required=True, help="path to the run's JSONL file")
    ap.add_argument(
        "--frame",
        choices=["raw", "corrected", "both"],
        default="corrected",
        help="which pose stream to score (default: corrected -- what's actually published)",
    )
    ap.add_argument(
        "--excursion-thresh-cm",
        type=float,
        default=5.0,
        help="deviation from resting baseline (cm) that counts as a disturbance event, "
        "not just ordinary noise (default: 5)",
    )
    ap.add_argument(
        "--min-gap-s",
        type=float,
        default=0.5,
        help="merge two over-threshold stretches into one event if the gap between them "
        "is shorter than this (default: 0.5s)",
    )
    ap.add_argument(
        "--baseline-window-s",
        type=float,
        default=5.0,
        help="how many seconds at the START and END of the run to use for the "
        "pre-/post-baseline residual-offset check (default: 5)",
    )
    ap.add_argument(
        "--stable-from-s",
        type=float,
        default=None,
        help="seconds since the run's first pose after which the camera was "
        "genuinely stationary (e.g. after you finished repositioning it to a "
        "better-tracked view). Fits a straight line to position-vs-time over "
        "just this window and reports the drift RATE (cm/s) -- the right "
        "metric for a slow monotonic creep during otherwise-healthy tracking, "
        "as opposed to the excursion-event detector above, which is aimed at "
        "a bounded disturbance (e.g. a hand wave) that comes back to baseline. "
        "If omitted, this section is skipped.",
    )
    args = ap.parse_args()

    frames_to_score = ["raw", "corrected"] if args.frame == "both" else [args.frame]
    thresh_m = args.excursion_thresh_cm / 100.0

    for frame in frames_to_score:
        poses = load_poses(args.jsonl, frame)
        if len(poses) < 2:
            print(f"[{frame}] no poses found for this frame -- skipping", file=sys.stderr)
            continue

        t0_ns = poses[0][0]
        positions = [p[1] for p in poses]
        baseline = median_position(positions)

        events = find_events(poses, baseline, thresh_m, args.min_gap_s)

        quiet_dists = [
            dist3(pos, baseline)
            for i, (t_ns, pos) in enumerate(poses)
            if not any(s <= i <= e for s, e in events)
        ]
        noise_floor_cm = stddev(quiet_dists) * 100.0

        print(f"\n=== frame={frame} ===")
        print(f"resting baseline (x,y,z) = ({baseline[0]:+.3f}, {baseline[1]:+.3f}, {baseline[2]:+.3f})")
        print(f"noise floor (quiet-segment stddev) = {noise_floor_cm:.2f}cm")
        print(f"{'event':<8}{'start_s':<10}{'end_s':<10}{'dur_s':<8}{'peak_excursion_cm':<20}")

        peak_excursions = []
        for idx, (s, e) in enumerate(events):
            seg = poses[s : e + 1]
            peak = max(dist3(pos, baseline) for _, pos in seg) * 100.0
            peak_excursions.append(peak)
            start_s = (poses[s][0] - t0_ns) / 1e9
            end_s = (poses[e][0] - t0_ns) / 1e9
            dur_s = end_s - start_s
            print(f"{idx:<8}{start_s:<10.2f}{end_s:<10.2f}{dur_s:<8.2f}{peak:<20.1f}")

        if not events:
            print("(no disturbance events detected above threshold)")

        # Residual offset: does the run end back where it started, or did
        # the corruption leave a permanent shift? Uses samples strictly
        # before the first event vs. the last baseline-window-s seconds of
        # the run, so a currently-active excursion at the very end doesn't
        # masquerade as "the new resting position".
        pre_end_idx = events[0][0] if events else len(poses)
        pre_positions = positions[:pre_end_idx] or positions[: max(1, len(positions) // 10)]
        pre_baseline = median_position(pre_positions)

        last_t_ns = poses[-1][0]
        post_positions = [
            pos for t_ns, pos in poses if (last_t_ns - t_ns) / 1e9 <= args.baseline_window_s
        ]
        post_baseline = median_position(post_positions)
        residual_offset_cm = dist3(pre_baseline, post_baseline) * 100.0

        max_peak = max(peak_excursions) if peak_excursions else 0.0
        mean_peak = sum(peak_excursions) / len(peak_excursions) if peak_excursions else 0.0

        print(
            f"\n[STATIONARY-HOLD SUMMARY] frame={frame} "
            f"num_events={len(events)} "
            f"max_peak_excursion_cm={max_peak:.1f} "
            f"mean_peak_excursion_cm={mean_peak:.1f} "
            f"noise_floor_cm={noise_floor_cm:.2f} "
            f"residual_offset_cm={residual_offset_cm:.1f}"
        )

        if args.stable_from_s is not None:
            window = [
                (t_ns, pos) for t_ns, pos in poses
                if (t_ns - t0_ns) / 1e9 >= args.stable_from_s
            ]
            if len(window) < 2:
                print(
                    f"\n[DRIFT-RATE] frame={frame} "
                    f"not enough samples after --stable-from-s={args.stable_from_s}s "
                    "-- skipping"
                )
            else:
                ts = [(t_ns - window[0][0]) / 1e9 for t_ns, _ in window]
                xs = [pos[0] for _, pos in window]
                ys = [pos[1] for _, pos in window]
                zs = [pos[2] for _, pos in window]

                slope_x = linear_fit_slope(ts, xs) * 100.0
                slope_y = linear_fit_slope(ts, ys) * 100.0
                slope_z = linear_fit_slope(ts, zs) * 100.0
                rate_cm_s = math.sqrt(slope_x**2 + slope_y**2 + slope_z**2)

                window_dur_s = ts[-1]
                endpoint_drift_cm = dist3(window[0][1], window[-1][1]) * 100.0

                print(f"\n=== drift-rate (frame={frame}, window={window_dur_s:.1f}s "
                      f"starting at t={args.stable_from_s:.1f}s) ===")
                print(f"  per-axis linear fit: x={slope_x:+.3f}cm/s y={slope_y:+.3f}cm/s z={slope_z:+.3f}cm/s")
                print(
                    f"\n[DRIFT-RATE SUMMARY] frame={frame} "
                    f"drift_rate_cm_per_s={rate_cm_s:.3f} "
                    f"projected_drift_over_window_cm={rate_cm_s * window_dur_s:.1f} "
                    f"endpoint_to_endpoint_cm={endpoint_drift_cm:.1f} "
                    f"window_s={window_dur_s:.1f}"
                )


if __name__ == "__main__":
    main()
