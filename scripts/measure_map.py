#!/usr/bin/env python3
"""Measure distances in a reconstructed map by picking points.

Opens the map in a window. Shift + left-click to pick points (Shift +
right-click undoes the last pick), then close the window (Q). Distances are
printed for each consecutive pair: 1-2, 3-4, 5-6, ... Map units are metres.

  ~/Documents/basalt/.venv_tsdf/bin/python ~/Documents/basalt/scripts/measure_map.py ~/scans/wall10/recon_pg/mesh.ply
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


def load(path):
    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.triangles):
        # Dense, even sampling of the surface is easier to click on than raw vertices.
        n = max(200_000, len(mesh.vertices) * 2)
        pcd = mesh.sample_points_uniformly(n)
    else:
        pcd = o3d.io.read_point_cloud(str(path))
    if not len(pcd.points):
        sys.exit(f"{path} has no points")
    if not pcd.has_colors():
        z = np.asarray(pcd.points)[:, 2]
        t = np.clip((z - z.min()) / max(np.ptp(z), 1e-6), 0, 1)
        pcd.colors = o3d.utility.Vector3dVector(np.c_[t, 0.4 + 0.4 * (1 - t), 1 - t])
    return pcd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map", type=Path, help="mesh.ply or points.ply")
    args = ap.parse_args()

    pcd = load(args.map)
    P = np.asarray(pcd.points)
    ext = np.ptp(P, 0)
    print(f"Map size (x, y, z extent): {ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f} m  (z is up)")
    print("Shift + left-click to pick points, Shift + right-click to undo, Q to finish.")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=f"measure: {args.map}", width=1400, height=900)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

    idx = vis.get_picked_points()
    if len(idx) < 2:
        print("Pick at least two points.")
        return
    print()
    for k in range(0, len(idx) - 1, 2):
        a, b = P[idx[k]], P[idx[k + 1]]
        d = b - a
        print(f"#{k + 1}-#{k + 2}: {np.linalg.norm(d):.3f} m   "
              f"(horizontal {np.linalg.norm(d[:2]):.3f} m, vertical {abs(d[2]):.3f} m)")
    if len(idx) % 2:
        print(f"(point #{len(idx)} has no partner and was ignored)")


if __name__ == "__main__":
    main()
