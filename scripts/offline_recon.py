#!/usr/bin/env python3
"""Drift-corrected offline reconstruction from a basalt_oak_d_vio --record-depth-dir session.

offline_tsdf.py fuses every depth frame at its VIO pose, so slow VIO position
drift places the same wall at different depths over the scan (layers). This
script removes that drift the way Open3D's reconstruction system does:

  1. Fragments: cut the scan into short chunks (default 4 s). VIO is accurate
     over a few seconds, so each chunk fuses into a clean local surface.
  2. Registration: align consecutive fragments, and any non-adjacent pair the
     VIO poses say overlaps (loop closures, e.g. revisiting the same wall),
     with multi-scale point-to-plane ICP started from the VIO guess.
  3. Pose graph: jointly solve all fragment poses so every alignment agrees
     (Open3D global optimization; bad loop matches are pruned). Weak VIO
     edges keep directions the depth can't constrain (sliding along a flat
     wall) at their VIO values.
  4. Fusion: per-frame world correction interpolated between fragment
     centers, then the same TSDF + multi-view consistency filter as
     offline_tsdf.py.

Run with the dedicated venv:
  ~/Documents/basalt/.venv_tsdf/bin/python ~/Documents/basalt/scripts/offline_recon.py SESSION_DIR [--publish]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from offline_tsdf import PoseInterpolator, export_surface, load_session, reject_isolated  # noqa: E402

reg = o3d.pipelines.registration


def rot_deg(T):
    return np.degrees(np.linalg.norm(Rotation.from_matrix(T[:3, :3]).as_rotvec()))


class Frames:
    """Depth frames with VIO camera poses and the same per-frame cleanup as offline_tsdf.py."""

    def __init__(self, args, meta, T_i_c, traj, frames):
        self.args, self.meta = args, meta
        self.w, self.h = int(meta["width"]), int(meta["height"])
        self.bx, self.by = int(self.w * args.border), int(self.h * args.border)
        self.intr = o3d.camera.PinholeCameraIntrinsic(self.w, self.h, meta["fx"], meta["fy"], meta["cx"], meta["cy"])
        self.gray = o3d.geometry.Image(np.full((self.h, self.w, 3), 160, np.uint8))
        interp = PoseInterpolator(traj, args.max_pose_gap_ms * 1e6)
        self.items = []  # (t_ns, file, T_w_c from VIO)
        for t_ns, name in frames[::args.every]:
            T_w_i = interp(t_ns)
            if T_w_i is not None:
                self.items.append((t_ns, name, T_w_i @ T_i_c))
        self.skipped = len(frames[::args.every]) - len(self.items)

    def rgbd(self, name):
        d = cv2.imread(str(self.args.session / "depth" / name), cv2.IMREAD_UNCHANGED)
        if d is None or d.shape != (self.h, self.w):
            return None, 0.0
        coverage = (d > 0).mean()
        d = d.astype(np.uint16)
        d[:self.by, :] = 0; d[self.h - self.by:, :] = 0; d[:, :self.bx] = 0; d[:, self.w - self.bx:] = 0
        d[d < self.args.min_depth * self.meta["depth_scale"]] = 0
        d = reject_isolated(d)
        img = o3d.geometry.RGBDImage.create_from_color_and_depth(
            self.gray, o3d.geometry.Image(d), depth_scale=self.meta["depth_scale"],
            depth_trunc=self.args.max_depth, convert_rgb_to_intensity=False)
        return img, coverage

    def volume(self):
        return o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.args.voxel, sdf_trunc=self.args.trunc or 4 * self.args.voxel,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)


def build_fragment(F, idx):
    """Fuse frames idx (VIO poses, relative to the first one) into a local cloud with normals."""
    T0_inv = np.linalg.inv(F.items[idx[0]][2])
    vol = F.volume()
    for i in idx:
        img, _ = F.rgbd(F.items[i][1])
        if img is not None:
            vol.integrate(img, F.intr, np.linalg.inv(T0_inv @ F.items[i][2]))
    pcd = vol.extract_point_cloud().voxel_down_sample(F.args.voxel * 1.5)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=F.args.voxel * 5, max_nn=30))
    return pcd


def point_to_plane_info(src, dst, T, corr):
    """6x6 information (rotation, translation order, as Open3D's pose graph
    expects) of point-to-plane residuals: each correspondence only constrains
    motion along the target normal, so a flat wall gives no information about
    sliding along itself -- unlike Open3D's point-to-point information."""
    if len(corr) == 0:
        return np.zeros((6, 6))
    p = (np.asarray(src.points)[corr[:, 0]] @ T[:3, :3].T) + T[:3, 3]
    n = np.asarray(dst.normals)[corr[:, 1]]
    J = np.hstack([np.cross(p, n), n])
    return J.T @ J


def register(src, dst, init, scales, inlier_dist=0.04):
    """Translation-only point-to-plane ICP from init (rotation kept at the VIO
    value). Full 6-DoF ICP between partially overlapping, cluttered real
    fragments returned 3-12 deg rotations over 4 s -- implausible for a gyro --
    i.e. it twisted onto the wrong surfaces; the drift that layers the map is
    in position. Returns (T, 6x6 translation information, fitness, rmse)."""
    from scipy.spatial import cKDTree
    Q, N = np.asarray(dst.points), np.asarray(dst.normals)
    tree = cKDTree(Q)
    P0 = np.asarray(src.points) @ init[:3, :3].T + init[:3, 3]
    t = np.zeros(3)
    for dist in scales:
        for _ in range(20):
            dd, ii = tree.query(P0 + t, distance_upper_bound=dist)
            ok = np.isfinite(dd)
            if ok.sum() < 100:
                break
            n = N[ii[ok]]
            r = ((P0[ok] + t - Q[ii[ok]]) * n).sum(1)
            w = np.where(np.abs(r) < dist, (1 - (r / dist) ** 2) ** 2, 0.0)  # Tukey
            A = (n * w[:, None]).T @ n
            b = -(n * (w * r)[:, None]).sum(0)
            ev, U = np.linalg.eigh(A)
            # Only step along directions the geometry constrains: point-to-plane gives
            # no information about sliding along a wall (or along two walls' meeting
            # line), and an unconstrained solve there slid ~30 cm in testing.
            strong = ev > 0.02 * max(ev.max(), 1e-12)
            step = U[:, strong] @ ((U[:, strong].T @ b) / ev[strong])
            t += step
            if np.linalg.norm(step) < 1e-4:
                break
    dd, ii = tree.query(P0 + t, distance_upper_bound=inlier_dist)
    ok = np.isfinite(dd)
    n = N[ii[ok]]
    info = np.zeros((6, 6))
    info[3:, 3:] = n.T @ n
    rmse = float(np.sqrt(np.mean(dd[ok] ** 2))) if ok.any() else np.inf
    T = init.copy()
    T[:3, 3] += t
    return T, info, float(ok.mean()), rmse


def overlap(src, dst, T, radius):
    """Fraction of src points (moved by T) within radius of dst."""
    s = o3d.geometry.PointCloud(src).transform(T)
    return float((np.asarray(s.compute_point_cloud_distance(dst)) < radius).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path)
    ap.add_argument("--voxel", type=float, default=0.02, help="TSDF voxel size (m)")
    ap.add_argument("--trunc", type=float, default=None, help="truncation distance (m), default 4x voxel")
    ap.add_argument("--min-depth", type=float, default=0.25)
    ap.add_argument("--max-depth", type=float, default=2.5)
    ap.add_argument("--border", type=float, default=0.12)
    ap.add_argument("--max-pose-gap-ms", type=float, default=100.0)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--min-support", type=int, default=3)
    ap.add_argument("--support-frames", type=int, default=300)
    ap.add_argument("--fragment-frames", type=int, default=40, help="depth frames per fragment (~4 s at 10 Hz)")
    ap.add_argument("--fit-min", type=float, default=0.3,
                    help="min fraction of a fragment's points aligned (within ~3 cm) to accept a registration")
    ap.add_argument("--vio-trans-sigma", type=float, default=0.05,
                    help="assumed VIO relative-translation error between consecutive fragments (m)")
    ap.add_argument("--vio-rot-sigma-deg", type=float, default=0.5,
                    help="assumed VIO (gyro) relative-rotation error between consecutive fragments (deg)")
    ap.add_argument("--max-loop-correction", type=float, default=1.0,
                    help="reject a loop closure that moves a fragment more than this far from its VIO guess (m)")
    ap.add_argument("--verbose", action="store_true", help="print every consecutive-fragment registration")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default SESSION/recon_pg)")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--dashboard", type=Path, default=Path.home() / "Documents/VIO_Dashboard")
    args = ap.parse_args()
    out = args.out or args.session / "recon_pg"
    out.mkdir(parents=True, exist_ok=True)
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    meta, T_i_c, traj, frames = load_session(args.session)
    F = Frames(args, meta, T_i_c, traj, frames)
    if len(F.items) < 2:
        sys.exit("No frames with a VIO pose -- check that trajectory.csv and depth timestamps overlap.")
    n = len(F.items)
    k = max(5, args.fragment_frames)
    frag_idx = [list(range(s, min(n, s + k))) for s in range(0, n, k)]
    if len(frag_idx) > 1 and len(frag_idx[-1]) < k // 2:  # fold a short tail into the previous fragment
        tail = frag_idx.pop()
        frag_idx[-1] += tail
    m = len(frag_idx)
    print(f"{n} frames ({F.skipped} without a pose) -> {m} fragments of ~{k} frames")

    # 1. Fragments
    clouds, P_vio = [], []
    for f, idx in enumerate(frag_idx):
        clouds.append(build_fragment(F, idx))
        P_vio.append(F.items[idx[0]][2])
        print(f"  fragment {f + 1}/{m}: {len(clouds[-1].points)} points", flush=True)

    # 2. Registration -> pose graph (node pose = fragment frame -> world)
    vio_info = np.diag([1 / np.radians(args.vio_rot_sigma_deg) ** 2] * 3 + [1 / args.vio_trans_sigma ** 2] * 3)
    pg = reg.PoseGraph()
    for P in P_vio:
        pg.nodes.append(reg.PoseGraphNode(P))
    odo_ok = loops_tried = loops_ok = 0
    fine = [0.16, 0.08, 0.04]
    for i in range(m):
        for j in range(i + 1, m):
            init = np.linalg.inv(P_vio[j]) @ P_vio[i]  # maps fragment i's frame into fragment j's
            if j == i + 1:
                T, info, fit, rmse = register(clouds[i], clouds[j], init, fine)
                good = fit > args.fit_min and np.linalg.norm(T[:3, 3] - init[:3, 3]) < 0.3
                if args.verbose:
                    print(f"    pair {i}-{j}: fitness {fit:.2f} rmse {rmse * 100:.1f} cm, moved "
                          f"{np.linalg.norm(T[:3, 3] - init[:3, 3]) * 100:.0f} cm"
                          f" -> {'ok' if good else 'REJECTED'}")
                if good:
                    odo_ok += 1
                else:
                    T, info = init, np.zeros((6, 6))
                pg.edges.append(reg.PoseGraphEdge(i, j, T, info + vio_info, uncertain=False))
                continue
            if overlap(clouds[i], clouds[j], init, 0.3) < 0.3:
                continue
            loops_tried += 1
            T, info, fit, rmse = register(clouds[i], clouds[j], init, [0.4, 0.2] + fine)
            moved = np.linalg.norm(T[:3, 3] - init[:3, 3])
            if fit > args.fit_min and moved < args.max_loop_correction:
                loops_ok += 1
                # Rotation of a loop edge is just the VIO guess: weak rotation information
                # (~5 deg) so it constrains position without fighting the gyro chain.
                info[:3, :3] = np.eye(3) / np.radians(5.0) ** 2
                pg.edges.append(reg.PoseGraphEdge(i, j, T, info + 1e-6 * np.eye(6), uncertain=True))
    print(f"Registration: {odo_ok}/{m - 1} consecutive pairs aligned, "
          f"{loops_ok}/{loops_tried} overlapping non-adjacent pairs accepted as loop closures")

    # 3. Global optimization
    reg.global_optimization(
        pg, reg.GlobalOptimizationLevenbergMarquardt(), reg.GlobalOptimizationConvergenceCriteria(),
        reg.GlobalOptimizationOption(max_correspondence_distance=0.05, edge_prune_threshold=0.25,
                                     preference_loop_closure=1.0, reference_node=0))
    C = [pg.nodes[f].pose @ np.linalg.inv(P_vio[f]) for f in range(m)]  # world-frame corrections
    np.savez(out / "fragments.npz", P_vio=np.array(P_vio), P_opt=np.array([nd.pose for nd in pg.nodes]),
             first_frame=np.array([F.items[idx[0]][0] for idx in frag_idx]))
    moves = [np.linalg.norm(c[:3, 3]) for c in C]
    print(f"Pose graph: {len(pg.edges)} edges kept; fragment corrections up to {max(moves) * 100:.0f} cm "
          f"(median {np.median(moves) * 100:.0f} cm), {max(rot_deg(c) for c in C):.1f} deg")

    # 4. Fusion with corrections interpolated between fragment centers
    centers = np.array([np.mean(idx) for idx in frag_idx])
    slerp = Slerp(centers, Rotation.from_matrix([c[:3, :3] for c in C])) if m > 1 else None
    trans = np.array([c[:3, 3] for c in C])
    vol = F.volume()
    posed, coverage = [], []
    for i, (_, name, T_vio) in enumerate(F.items):
        img, cov = F.rgbd(name)
        if img is None:
            continue
        coverage.append(cov)
        corr = np.eye(4)
        if slerp is not None:
            x = float(np.clip(i, centers[0], centers[-1]))
            corr[:3, :3] = slerp([x]).as_matrix()[0]
            corr[:3, 3] = [np.interp(x, centers, trans[:, a]) for a in range(3)]
        else:
            corr = C[0]
        T_w_c = corr @ T_vio
        vol.integrate(img, F.intr, np.linalg.inv(T_w_c))
        posed.append((name, T_w_c))
    print(f"Fused {len(posed)} frames; depth coverage median {np.median(coverage) * 100:.0f}% of pixels")
    export_surface(vol.extract_triangle_mesh(), posed, args, meta, F.bx, F.by, out, label="Recon")


if __name__ == "__main__":
    main()
