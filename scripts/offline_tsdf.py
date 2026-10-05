#!/usr/bin/env python3
"""Offline TSDF reconstruction from a basalt_oak_d_vio --record-depth-dir session.

Fuses every recorded depth frame into one truncated signed distance field
(Open3D ScalableTSDFVolume) using the raw VIO trajectory interpolated at each
depth timestamp, optionally refined frame-to-model with point-to-plane ICP,
then exports a mesh and a point cloud. A TSDF averages repeated observations
of the same surface into one thin zero crossing, instead of accumulating
every noisy return as a permanently occupied voxel the way the live octomap
does.

Run with the dedicated venv:
  ~/Documents/basalt/.venv_tsdf/bin/python ~/Documents/basalt/scripts/offline_tsdf.py SESSION_DIR [--icp] [--publish]
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation, Slerp


def se3(t, q_xyzw):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(q_xyzw).as_matrix()
    T[:3, 3] = t
    return T


def load_session(d):
    meta = json.loads((d / "meta.json").read_text())
    tic = meta["T_i_c"]
    T_i_c = se3([tic["px"], tic["py"], tic["pz"]],
                [tic["qx"], tic["qy"], tic["qz"], tic["qw"]])
    traj = np.loadtxt(d / "trajectory.csv", delimiter=",", skiprows=1, ndmin=2)
    if len(traj) < 2:
        sys.exit(f"{d/'trajectory.csv'} has no poses -- the VIO run ended before writing it "
                 "(older builds only flushed on a clean exit). Record the session again.")
    traj = traj[np.argsort(traj[:, 0])]
    _, uniq = np.unique(traj[:, 0], return_index=True)
    traj = traj[uniq]
    with open(d / "frames.csv") as f:
        frames = [(int(r["t_ns"]), r["file"]) for r in csv.DictReader(f)]
    return meta, T_i_c, traj, sorted(frames)


class PoseInterpolator:
    def __init__(self, traj, max_gap_ns):
        self.t = traj[:, 0].astype(np.int64)
        self.p = traj[:, 1:4]
        self.slerp = Slerp(self.t.astype(np.float64), Rotation.from_quat(traj[:, 4:8]))
        self.max_gap = max_gap_ns

    def __call__(self, t_ns):
        i = np.searchsorted(self.t, t_ns)
        if i == 0 or i >= len(self.t):
            return None
        t0, t1 = self.t[i - 1], self.t[i]
        if t1 - t0 > self.max_gap:
            return None
        a = (t_ns - t0) / (t1 - t0)
        T = np.eye(4)
        T[:3, :3] = self.slerp([float(t_ns)]).as_matrix()[0]
        T[:3, 3] = (1 - a) * self.p[i - 1] + a * self.p[i]
        return T


def reject_isolated(d, rel=0.04, abs_mm=30):
    """Zero depth pixels that disagree with their 5x5 neighborhood median
    (speckles and flying pixels at depth edges)."""
    import cv2
    med = cv2.medianBlur(d, 5)
    tol = np.maximum(abs_mm, rel * med.astype(np.float32))
    bad = (med == 0) | (np.abs(d.astype(np.float32) - med) > tol)
    out = d.copy()
    out[bad] = 0
    return out


def count_support(verts, posed, args, meta, bx, by):
    """For each vertex, count frames whose measured depth at the vertex's
    projection agrees with the vertex's own depth (within ~2 voxels)."""
    import cv2
    w, h = int(meta["width"]), int(meta["height"])
    fx, fy, cx, cy = meta["fx"], meta["fy"], meta["cx"], meta["cy"]
    tol = max(0.03, 2 * args.voxel)
    step = max(1, len(posed) // args.support_frames)
    support = np.zeros(len(verts), np.int32)
    vh = np.c_[verts, np.ones(len(verts))]
    for name, T_w_c in posed[::step]:
        d = cv2.imread(str(args.session / "depth" / name), cv2.IMREAD_UNCHANGED)
        pc = (np.linalg.inv(T_w_c) @ vh.T)[:3].T
        z = pc[:, 2]
        ok = z > 0.05
        u = np.full(len(z), -1, np.int64); v = u.copy()
        u[ok] = np.round(fx * pc[ok, 0] / z[ok] + cx).astype(np.int64)
        v[ok] = np.round(fy * pc[ok, 1] / z[ok] + cy).astype(np.int64)
        ok &= (u >= bx) & (u < w - bx) & (v >= by) & (v < h - by)
        zm = np.zeros(len(z))
        zm[ok] = d[v[ok], u[ok]] / meta["depth_scale"]
        ok &= zm > 0
        support[ok & (np.abs(zm - z) < tol)] += 1
    return support


def export_surface(mesh, posed, args, meta, bx, by, out, label="TSDF"):
    """Multi-view consistency filter + fragment cleanup on a fused mesh, then
    write mesh.ply / height-colored points.ply (and optionally publish)."""
    verts = np.asarray(mesh.vertices)
    support = count_support(verts, posed, args, meta, bx, by)
    keep = support >= args.min_support
    print(f"Multi-view consistency: kept {keep.mean()*100:.1f}% of {len(verts)} surface vertices "
          f"(support >= {args.min_support} frames)")
    mesh.remove_vertices_by_mask(~keep)
    # Drop tiny disconnected fragments (stray noise blobs).
    tri_clusters, cluster_n, _ = mesh.cluster_connected_triangles()
    tri_clusters, cluster_n = np.asarray(tri_clusters), np.asarray(cluster_n)
    mesh.remove_triangles_by_mask(cluster_n[tri_clusters] < 200)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out / "mesh.ply"), mesh)

    pcd = o3d.geometry.PointCloud(mesh.vertices)
    z = np.asarray(pcd.points)[:, 2]
    tnorm = np.clip((z - np.percentile(z, 2)) / max(1e-6, np.ptp(np.percentile(z, [2, 98]))), 0, 1)
    colors = np.stack([tnorm, 0.35 + 0.5 * (1 - np.abs(tnorm - 0.5) * 2), 1 - tnorm], axis=1)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(out / "points.ply"), pcd)
    print(f"Wrote {out/'mesh.ply'} ({len(mesh.triangles)} triangles) and {out/'points.ply'} ({len(pcd.points)} points)")

    if args.publish:
        sys.path.insert(0, str(args.dashboard / "backend"))
        from app.maps import MapStorage
        meta_out = MapStorage(args.dashboard / "backend/data/maps").save(
            run_id=args.session.name, source_path=out / "points.ply",
            name=f"{label} {args.session.name}")
        print(f"Published to dashboard MAPS tab as {meta_out['map_id']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path)
    ap.add_argument("--voxel", type=float, default=0.02, help="TSDF voxel size (m)")
    ap.add_argument("--trunc", type=float, default=None, help="truncation distance (m), default 4x voxel")
    ap.add_argument("--min-depth", type=float, default=0.25)
    ap.add_argument("--max-depth", type=float, default=3.0)
    ap.add_argument("--border", type=float, default=0.12,
                    help="fraction of image width/height dropped on each side (noisy wide-FOV edges)")
    ap.add_argument("--max-pose-gap-ms", type=float, default=100.0,
                    help="skip a frame if the VIO samples around it are further apart than this")
    ap.add_argument("--every", type=int, default=1, help="use every Nth depth frame")
    ap.add_argument("--min-support", type=int, default=3,
                    help="keep a surface point only if at least this many frames measured a surface there")
    ap.add_argument("--support-frames", type=int, default=300,
                    help="max frames (evenly sampled) used for the consistency check")
    ap.add_argument("--icp", action="store_true", help="refine poses frame-to-model with point-to-plane ICP")
    ap.add_argument("--rotation-only", action="store_true",
                    help="use only the VIO rotation (gyro-backed, reliable) and estimate position from depth "
                         "with frame-to-model ICP; for scans where VIO position drifts. Implies --icp")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default SESSION/recon)")
    ap.add_argument("--publish", action="store_true", help="copy the point cloud into the dashboard MAPS tab")
    ap.add_argument("--dashboard", type=Path, default=Path.home() / "Documents/VIO_Dashboard")
    args = ap.parse_args()

    args.icp = args.icp or args.rotation_only
    trunc = args.trunc or 4 * args.voxel
    out = args.out or args.session / "recon"
    out.mkdir(parents=True, exist_ok=True)

    meta, T_i_c, traj, frames = load_session(args.session)
    w, h = int(meta["width"]), int(meta["height"])
    intr = o3d.camera.PinholeCameraIntrinsic(w, h, meta["fx"], meta["fy"], meta["cx"], meta["cy"])
    interp = PoseInterpolator(traj, args.max_pose_gap_ms * 1e6)
    bx, by = int(w * args.border), int(h * args.border)
    gray = o3d.geometry.Image(np.full((h, w, 3), 160, np.uint8))

    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel, sdf_trunc=trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)

    correction = np.eye(4)  # world-frame drift correction found by ICP
    position = None  # --rotation-only: camera position carried forward from the last ICP-refined frame
    model = None
    used = skipped = icp_ok = 0
    posed = []  # (depth file, T_w_c actually used) for the consistency pass
    coverage = []  # fraction of each raw frame with any depth
    for k, (t_ns, name) in enumerate(frames[::args.every]):
        T_w_i = interp(t_ns)
        if T_w_i is None:
            skipped += 1
            continue
        depth = o3d.io.read_image(str(args.session / "depth" / name))
        d = np.asarray(depth).astype(np.uint16)
        if d.shape != (h, w):
            skipped += 1
            continue
        d = d.copy()
        coverage.append((d > 0).mean())
        d[:by, :] = 0; d[h - by:, :] = 0; d[:, :bx] = 0; d[:, w - bx:] = 0
        d[d < args.min_depth * meta["depth_scale"]] = 0
        d = reject_isolated(d)
        depth = o3d.geometry.Image(d)
        T_w_c = correction @ T_w_i @ T_i_c
        if args.rotation_only:
            if position is None:
                position = T_w_c[:3, 3].copy()
            T_w_c[:3, 3] = position

        if args.icp and model is not None and len(model.points) >= 500:
            frame_pcd = o3d.geometry.PointCloud.create_from_depth_image(
                depth, intr, np.linalg.inv(T_w_c), depth_scale=meta["depth_scale"],
                depth_trunc=args.max_depth, stride=4)
            frame_pcd = frame_pcd.voxel_down_sample(args.voxel * 2)
            if len(frame_pcd.points) > 200:
                reg = o3d.pipelines.registration.registration_icp(
                    frame_pcd, model, 3 * args.voxel, np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
                step = np.linalg.norm(reg.transformation[:3, 3])
                angle = np.degrees(np.linalg.norm(Rotation.from_matrix(reg.transformation[:3, :3]).as_rotvec()))
                min_fit = 0.3 if args.rotation_only else 0.5
                if reg.fitness > min_fit and step < 0.10 and angle < 5.0:
                    T_w_c = reg.transformation @ T_w_c
                    if args.rotation_only:
                        # Keep only the rotation part of the correction; the position is tracked directly.
                        correction[:3, :3] = reg.transformation[:3, :3] @ correction[:3, :3]
                    else:
                        correction = reg.transformation @ correction
                    icp_ok += 1
        if args.rotation_only:
            position = T_w_c[:3, 3].copy()

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            gray, depth, depth_scale=meta["depth_scale"], depth_trunc=args.max_depth,
            convert_rgb_to_intensity=False)
        vol.integrate(rgbd, intr, np.linalg.inv(T_w_c))
        used += 1
        posed.append((name, T_w_c))

        if args.icp and (model is None or len(model.points) < 500 or used % 10 == 0):
            model = vol.extract_point_cloud().voxel_down_sample(args.voxel * 2)
            model.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4 * args.voxel, max_nn=30))
        if used % 50 == 0:
            print(f"  integrated {used} frames ({skipped} skipped)", flush=True)

    print(f"Integrated {used} frames, skipped {skipped} (no pose within gap or bad size)"
          + (f", ICP accepted on {icp_ok}" if args.icp else ""))
    if coverage:
        cov = np.median(coverage) * 100
        print(f"Depth coverage: median {cov:.0f}% of pixels per frame"
              + ("  <-- too sparse for a clean surface (expect >40%); check the stereo"
                 " confidence threshold, IR projector and scanning distance" if cov < 25 else ""))
    if used == 0:
        sys.exit("No frames integrated -- check that trajectory.csv and depth timestamps overlap.")

    export_surface(vol.extract_triangle_mesh(), posed, args, meta, bx, by, out)


if __name__ == "__main__":
    main()
