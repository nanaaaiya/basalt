#!/usr/bin/env python3
"""Localize a basalt_oak_d_vio session against a saved map (map-based relocalization).

Follows the session's --record-depth-dir folder while it is being written
(live) or replays a finished one (--replay), and keeps one correction
T_map_odom between this flight's VIO world frame and the saved map's frame:

  * Start-up: the first ~2 s of depth (fused with VIO poses) are matched to
    the map from the takeoff assumption "same spot as the mapping flight"
    (identity), trying headings within +---init-yaw-range deg.
  * Tracking: every --update-period s the last --window s of depth are
    matched again from the current estimate; the correction is updated if
    the match is good and the step is plausible, otherwise counted as a
    miss (--max-misses in a row -> "lost" -> start-up search again).

Both VIO world and map frames are gravity-aligned, so matching solves only
x, y, z and yaw (point-to-plane ICP, Tukey-robust, skipping directions the
geometry can't observe -- e.g. sliding along a single flat wall).

Every VIO pose is republished as T_map_odom * pose (frame "map", ~30 Hz) to
the VIO Dashboard, plus a "localization" status after each match. The
active map is chosen in the dashboard's MAPS tab (or with --map).

Run with the dedicated venv:
  ~/Documents/basalt/.venv_tsdf/bin/python ~/Documents/basalt/scripts/localizer.py SESSION_DIR
"""

import argparse
import csv
import json
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from offline_tsdf import reject_isolated, se3  # noqa: E402


def yaw_deg(T):
    return float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))


class Map:
    def __init__(self, path, voxel):
        pcd = o3d.io.read_point_cloud(str(path)).voxel_down_sample(voxel)
        if len(pcd.points) < 100:
            raise ValueError(f"map {path} has too few points ({len(pcd.points)})")
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=30))
        self.Q = np.asarray(pcd.points)
        self.N = np.asarray(pcd.normals)
        self.tree = cKDTree(self.Q)


def icp_4dof(P, m, T0, scales, iters=15, inlier=0.05):
    """Point-to-plane ICP of odom-frame points P onto map m, solving x, y, z, yaw.
    Returns (T_map_odom, fitness = fraction of points within `inlier`, rmse)."""
    T = T0.copy()
    for dist in scales:
        for _ in range(iters):
            X = P @ T[:3, :3].T + T[:3, 3]
            dd, ii = m.tree.query(X, distance_upper_bound=dist)
            ok = np.isfinite(dd)
            if ok.sum() < 50:
                break
            x, q, n = X[ok], m.Q[ii[ok]], m.N[ii[ok]]
            c = x.mean(0)
            r = ((x - q) * n).sum(1)
            w = np.where(np.abs(r) < dist, (1 - (r / dist) ** 2) ** 2, 0.0)
            xc = x - c
            J = np.c_[n, n[:, 1] * xc[:, 0] - n[:, 0] * xc[:, 1]]  # d r / d(tx, ty, tz, yaw about c)
            A = (J * w[:, None]).T @ J
            b = -(J * (w * r)[:, None]).sum(0)
            ev, U = np.linalg.eigh(A)
            strong = ev > 0.02 * max(ev.max(), 1e-12)
            dx = U[:, strong] @ ((U[:, strong].T @ b) / ev[strong])
            Rz = Rotation.from_euler("z", dx[3]).as_matrix()
            D = np.eye(4)
            D[:3, :3] = Rz
            D[:3, 3] = c - Rz @ c + dx[:3]
            T = D @ T
            if np.linalg.norm(dx[:3]) < 1e-4 and abs(dx[3]) < 1e-5:
                break
    X = P @ T[:3, :3].T + T[:3, 3]
    dd, _ = m.tree.query(X, distance_upper_bound=inlier)
    ok = np.isfinite(dd)
    rmse = float(np.sqrt(np.mean(dd[ok] ** 2))) if ok.any() else float("inf")
    return T, float(ok.mean()), rmse


class Session:
    """A --record-depth-dir folder, followed while it grows (poll()) or read whole."""

    def __init__(self, d, wait=True):
        self.d = d
        while not (d / "meta.json").exists():
            if not wait:
                sys.exit(f"{d}/meta.json not found")
            time.sleep(0.2)
        time.sleep(0.2)
        self.meta = json.loads((d / "meta.json").read_text())
        tic = self.meta["T_i_c"]
        self.T_i_c = se3([tic["px"], tic["py"], tic["pz"]], [tic["qx"], tic["qy"], tic["qz"], tic["qw"]])
        self.traj = []    # (t_ns, 7 floats)
        self.frames = []  # (t_ns, file)
        self._files = {}

    def _new_lines(self, name):
        f = self._files.get(name)
        if f is None:
            p = self.d / name
            if not p.exists():
                return []
            f = self._files[name] = [open(p), ""]
            f[0].readline()  # header
        f[1] += f[0].read()
        *lines, f[1] = f[1].split("\n")
        return [ln for ln in lines if ln]

    def poll(self):
        new_poses = []
        for ln in self._new_lines("trajectory.csv"):
            v = ln.split(",")
            new_poses.append((int(v[0]), [float(x) for x in v[1:8]]))
        for ln in self._new_lines("frames.csv"):
            t, name = ln.split(",")
            self.frames.append((int(t), name))
        self.traj.extend(new_poses)
        return new_poses


def pose_matrix(vals):
    return se3(vals[:3], vals[3:7])


class Localizer:
    def __init__(self, session, args):
        self.s, self.a = session, args
        m = session.meta
        self.fx, self.fy, self.cx, self.cy = m["fx"], m["fy"], m["cx"], m["cy"]
        self.w, self.h = int(m["width"]), int(m["height"])
        self.stride = max(1, self.w // 160)
        self.map = None
        self.map_id = None
        self.T = np.eye(4)  # T_map_odom
        self.state = "no_map"
        self.misses = 0
        self.last = {}
        self._cache = {}
        self.lock = threading.Lock()

    def set_map(self, path, map_id):
        self.map = Map(path, self.a.map_voxel)
        self.map_id = map_id
        self.T = np.eye(4)
        self.state = "initializing"
        self.misses = 0

    def _frame_points(self, t_ns, name, T_w_c):
        key = name
        if key not in self._cache:
            d = cv2.imread(str(self.s.d / "depth" / name), cv2.IMREAD_UNCHANGED)
            if d is None or d.shape != (self.h, self.w):
                return None
            bx, by = int(self.w * 0.12), int(self.h * 0.12)
            d = reject_isolated(d.astype(np.uint16))
            sub = d[by:self.h - by:self.stride, bx:self.w - bx:self.stride].astype(np.float32) / self.s.meta["depth_scale"]
            vv, uu = np.nonzero((sub > self.a.min_depth) & (sub < self.a.max_depth))
            z = sub[vv, uu]
            u = uu * self.stride + bx
            v = vv * self.stride + by
            self._cache[key] = np.c_[(u - self.cx) * z / self.fx, (v - self.cy) * z / self.fy, z]
            if len(self._cache) > 200:
                self._cache.pop(next(iter(self._cache)))
        pc = self._cache[key]
        return pc @ T_w_c[:3, :3].T + T_w_c[:3, 3]

    def local_cloud(self, now_ns):
        """Last --window s of depth, fused in the odom (VIO world) frame with VIO poses."""
        traj = self.s.traj
        if len(traj) < 2:
            return None
        t = np.array([p[0] for p in traj], np.int64)
        lo = now_ns - int(self.a.window * 1e9)
        sel = [(tf, n) for tf, n in self.s.frames if lo <= tf <= now_ns and t[0] < tf < t[-1]]
        if not sel:
            return None
        sel = sel[:: max(1, len(sel) // self.a.window_frames)]
        vals = np.array([p[1] for p in traj])
        slerp = Slerp(t.astype(np.float64), Rotation.from_quat(vals[:, 3:7]))
        pts = []
        for tf, name in sel:
            T_w_i = np.eye(4)
            T_w_i[:3, :3] = slerp([float(tf)]).as_matrix()[0]
            T_w_i[:3, 3] = [np.interp(tf, t, vals[:, k]) for k in range(3)]
            p = self._frame_points(tf, name, T_w_i @ self.s.T_i_c)
            if p is not None and len(p):
                pts.append(p)
        if not pts:
            return None
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack(pts)))
        return np.asarray(pcd.voxel_down_sample(self.a.cloud_voxel).points)

    def update(self, now_ns):
        """One matching step. Returns a status dict."""
        if self.map is None:
            return self._status(now_ns, "no_map")
        P = self.local_cloud(now_ns)
        if P is None or len(P) < 300:
            return self._status(now_ns, self.state, note="waiting for depth")
        with self.lock:
            T0 = self.T.copy()
        if self.state in ("initializing", "lost"):
            best = None
            c = P.mean(0)
            yaw_range = self.a.init_yaw_range
            for dy in np.arange(-yaw_range, yaw_range + 1e-6, 10.0):
                Rz = Rotation.from_euler("z", dy, degrees=True).as_matrix()
                D = np.eye(4)
                D[:3, :3] = Rz
                D[:3, 3] = c - Rz @ c
                T, fit, rmse = icp_4dof(P, self.map, D @ T0, [0.5, 0.25, 0.12, 0.06, 0.04])
                if best is None or fit > best[1]:
                    best = (T, fit, rmse)
            T, fit, rmse = best
            ok = fit >= self.a.init_fitness
            if ok:
                with self.lock:
                    self.T = T
                self.state, self.misses = "localized", 0
            return self._status(now_ns, self.state, fit, rmse)
        T, fit, rmse = icp_4dof(P, self.map, T0, [0.15, 0.08, 0.04])
        step = np.linalg.norm(T[:3, 3] - T0[:3, 3])
        dyaw = abs((yaw_deg(T) - yaw_deg(T0) + 180) % 360 - 180)
        if fit >= self.a.track_fitness and step < self.a.max_step and dyaw < self.a.max_step_yaw:
            with self.lock:
                self.T = T
            self.misses = 0
            self.state = "localized"
        else:
            self.misses += 1
            if self.misses >= self.a.max_misses:
                self.state = "lost"
        return self._status(now_ns, self.state, fit, rmse)

    def _status(self, now_ns, state, fit=None, rmse=None, note=None):
        with self.lock:
            T = self.T.copy()
        st = {"type": "localization", "source": "localizer", "t_ns": int(now_ns), "map_id": self.map_id,
              "state": state, "fitness": fit, "rmse": rmse, "misses": self.misses,
              "correction": {"x": float(T[0, 3]), "y": float(T[1, 3]), "z": float(T[2, 3]), "yaw_deg": yaw_deg(T)}}
        if note:
            st["note"] = note
        self.last = st
        return st

    def map_pose(self, t_ns, vals):
        with self.lock:
            T = self.T @ pose_matrix(vals)
        q = Rotation.from_matrix(T[:3, :3]).as_quat()
        return {"type": "pose", "source": "localizer", "t_ns": int(t_ns), "frame": "map",
                "position": [float(x) for x in T[:3, 3]], "orientation": [float(x) for x in q]}


class Backend:
    """Dashboard connection: active-map lookups over REST, messages over a WebSocket."""

    def __init__(self, url):
        self.url = url.rstrip("/")
        self.ws = None

    def send(self, msg):
        import websocket  # websocket-client
        try:
            if self.ws is None:
                self.ws = websocket.create_connection(self.url.replace("http", "ws", 1) + "/ws/localizer", timeout=2)
            self.ws.send(json.dumps(msg))
        except Exception:
            self.ws = None  # reconnect on the next message

    def active_map(self):
        try:
            with urllib.request.urlopen(self.url + "/localization", timeout=2) as r:
                return json.load(r).get("map_id")
        except Exception:
            return None

    def fetch_map(self, map_id):
        path = Path(tempfile.gettempdir()) / f"localizer_map_{map_id}.ply"
        if not path.exists():
            urllib.request.urlretrieve(f"{self.url}/maps/{map_id}/file", path)
        return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path, help="the --record-depth-dir of the flight to localize")
    ap.add_argument("--map", type=Path, default=None,
                    help="map .ply to localize against (default: the map chosen in the dashboard MAPS tab)")
    ap.add_argument("--backend", default="http://127.0.0.1:8000", help="VIO Dashboard backend URL")
    ap.add_argument("--no-backend", action="store_true", help="don't talk to the dashboard (offline testing)")
    ap.add_argument("--replay", action="store_true", help="process a finished recording instead of following a live one")
    ap.add_argument("--speed", type=float, default=0.0, help="replay pacing (1 = real time, 0 = as fast as possible)")
    ap.add_argument("--update-period", type=float, default=1.0, help="seconds between map matches")
    ap.add_argument("--window", type=float, default=2.0, help="seconds of recent depth matched each time")
    ap.add_argument("--window-frames", type=int, default=10, help="max depth frames used per match")
    ap.add_argument("--min-depth", type=float, default=0.3)
    ap.add_argument("--max-depth", type=float, default=3.0)
    ap.add_argument("--map-voxel", type=float, default=0.03)
    ap.add_argument("--cloud-voxel", type=float, default=0.04)
    ap.add_argument("--init-yaw-range", type=float, default=30.0, help="start-up heading search, +- deg")
    ap.add_argument("--init-fitness", type=float, default=0.5, help="fraction of points within 5 cm to accept start-up")
    ap.add_argument("--track-fitness", type=float, default=0.4)
    ap.add_argument("--max-step", type=float, default=0.3, help="reject a correction change larger than this (m)")
    ap.add_argument("--max-step-yaw", type=float, default=10.0, help="... or than this (deg)")
    ap.add_argument("--max-misses", type=int, default=5, help="failed matches in a row before 'lost'")
    args = ap.parse_args()

    backend = None if args.no_backend else Backend(args.backend)
    session = Session(args.session, wait=not args.replay)
    loc = Localizer(session, args)

    def check_map():
        if args.map is not None:
            if loc.map is None:
                loc.set_map(args.map, args.map.stem)
                print(f"Map: {args.map}")
            return
        map_id = backend.active_map() if backend else None
        if map_id and map_id != loc.map_id:
            loc.set_map(backend.fetch_map(map_id), map_id)
            print(f"Map: {map_id} (from dashboard)")
        elif not map_id and loc.map is not None:
            loc.map, loc.map_id, loc.state = None, None, "no_map"

    def report(st):
        if backend:
            backend.send(st)
        f = st["fitness"]
        c = st["correction"]
        print(f"  {st['state']:12s} fit {f if f is None else round(f, 2)}  correction "
              f"x {c['x']:+.2f} y {c['y']:+.2f} z {c['z']:+.2f} m, yaw {c['yaw_deg']:+.1f} deg", flush=True)

    log = open(args.session / "localized_trajectory.csv", "w")
    log.write("t_ns,tx,ty,tz,qx,qy,qz,qw,state\n")

    def emit(t_ns, vals):
        msg = loc.map_pose(t_ns, vals)
        if backend:
            backend.send(msg)
        log.write(f"{t_ns},{','.join(f'{x:.6f}' for x in msg['position'] + msg['orientation'])},{loc.state}\n")

    check_map()
    if args.replay:
        session.poll()
        poses, frames = session.traj, session.frames
        session.traj, session.frames = [], []
        fi, next_update, wall0, t0 = 0, None, time.time(), poses[0][0]
        for t_ns, vals in poses:
            while fi < len(frames) and frames[fi][0] <= t_ns:
                session.frames.append(frames[fi])
                fi += 1
            session.traj.append((t_ns, vals))
            if args.speed > 0:
                time.sleep(max(0.0, (t_ns - t0) / 1e9 / args.speed - (time.time() - wall0)))
            if next_update is None or t_ns >= next_update:
                next_update = t_ns + int(args.update_period * 1e9)
                report(loc.update(t_ns))
            emit(t_ns, vals)
    else:
        print(f"Following {args.session} (Ctrl+C to stop)")
        busy = threading.Event()

        def run_update(now):
            try:
                report(loc.update(now))
            finally:
                busy.clear()

        last_update = last_map_check = 0.0
        try:
            while True:
                for t_ns, vals in session.poll():
                    emit(t_ns, vals)
                now = time.time()
                if now - last_map_check > 1.0:
                    last_map_check = now
                    check_map()
                if now - last_update >= args.update_period and not busy.is_set() and session.traj:
                    last_update = now
                    busy.set()
                    threading.Thread(target=run_update, args=(session.traj[-1][0],), daemon=True).start()
                log.flush()
                time.sleep(0.02)
        except KeyboardInterrupt:
            pass
    log.close()


if __name__ == "__main__":
    main()
