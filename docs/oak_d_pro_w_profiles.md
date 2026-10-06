# OAK-D Pro W launch profiles

Two VIO settings profiles for `basalt_oak_d_vio` with the OAK-D Pro W. They
differ only in feature-tracking cost; everything else is shared.

| Profile | Config file | Use on |
|---|---|---|
| Full | `data/oak_d_pro_w_config.json` | Laptop-class CPUs (reference profile, e.g. for the Qualcomm board) |
| Pi 5 | `data/oak_d_pro_w_pi_config.json` | Raspberry Pi 5 |

The Pi 5 profile changes three values from the full profile:

| Setting | Full | Pi 5 |
|---|---|---|
| `optical_flow_detection_grid_size` | 35 | 50 |
| `optical_flow_levels` | 5 | 3 |
| `optical_flow_stereo_seed_depths_m` | 0.3-6.0 (CLI: 0.3 1.0 2.5 6.0) | 0.6, 2.0 |

With the full profile the Pi 5 runs VIO at ~16-23 poses/s with frequent
gaps; with the Pi 5 profile it holds ~30 poses/s with loop closure on.
Loop closure, depth saving and the live voxel map run at a lower CPU
priority than tracking, so they use only leftover CPU.

## Full profile (laptop)

```bash
./basalt_oak_d_vio \
  --cam-calib calib_results/calibration_oak_d_pro_w.json \
  --config-path data/oak_d_pro_w_config.json \
  --stereo-seed-depths 0.3 1.0 2.5 6.0 --online-loop-closure true \
  --enable-ir-emitters false \
  --occupancy-rate-hz 10 --record-depth-dir <new folder> \
  --dashboard-host <dashboard ip> --dashboard-port 8765
```

Depth defaults to full sensor resolution (1280x800 at 10 fps;
`--depth-full-res true`).

## Pi 5 profile

```bash
./basalt_oak_d_vio \
  --cam-calib calib_results/calibration_oak_d_pro_w.json \
  --config-path data/oak_d_pro_w_pi_config.json \
  --online-loop-closure true --show-gui false \
  --enable-ir-emitters false \
  --occupancy-rate-hz 10 --record-depth-dir <new folder> \
  --dashboard-host <dashboard ip> --dashboard-port 8765
```

Don't pass `--stereo-seed-depths` here: it would override the profile's
seed depths. The camera must be on a USB 3 port (check for 5000 Mbps); depth
runs at full resolution (1280x800) as on the laptop. The dashboard's Run button (scripts/dashboard_agent.py)
launches this profile and records each run to `~/scans/run_<timestamp>`.

## Shared settings

- Mono exposure is capped at 8 ms (`MAX_EXPOSURE_US` in `oak_d.h`) to limit
  motion blur.
- The IR dot projector stays off: its dots move with the camera and break
  tracking at close range.
- Use a new `--record-depth-dir` folder per run; an existing one is
  overwritten.
- Offline mapping: `scripts/offline_recon.py <recording> --publish` (see
  `scripts/requirements_tsdf.txt` for its Python environment).
