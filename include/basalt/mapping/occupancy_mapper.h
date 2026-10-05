/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Fuses depth frames (from OakDDevice's optional StereoDepth output) plus
// the pose each was captured at into a persistent 3D occupancy grid, using
// octomap -- see the research/proposal discussion this implements for why
// octomap specifically (a real, already-vendored, non-ROS dependency; see
// VIO_Dashboard's proposed "Occupancy Grid" panel for where the output is
// headed).
//
// Deliberately a separate module from OnlineLoopClosure, mirroring how
// ModalAI's own stack splits VIO (voxl-qvio-server) from mapping
// (voxl-mapper) -- this class only ever consumes a pose, it never produces
// one. The caller (oak_d_vio.cpp) is responsible for pairing each depth
// frame with whatever pose it wants mapped against (raw or
// loop-closure-corrected) before calling addDepthFrame(); this class has
// no opinion on which.
//
// Threading model matches DashboardClient/OnlineLoopClosure: addDepthFrame
// is called from a hot device-IO thread and must never block, so it only
// ever try_pushes onto a bounded queue and returns immediately, dropping
// the frame if the processing thread is still busy with a previous one
// (occupancy mapping doesn't need every single depth frame -- unlike VIO,
// missing one costs nothing but a slightly staler map). A dedicated
// processing thread owns the actual octree work.
//
// KNOWN LIMITATION, not caused by this class: the Pi5's OAK-D connection
// has a pre-existing hardware/USB instability (seen in earlier sessions
// independent of any mapping code) where the physical device can crash
// and reconnect mid-run (depthai logs "Device ... has crashed" then
// "Reconnection successful"). Confirmed via a live test with
// --enable-occupancy-mapping OFF that this reproduces with zero mapping
// code running at all -- so it's a device/environment issue, not a bug
// here. It does mean: (a) when the connection is healthy, this class's
// own logic is verified correct (both synthetically -- see
// test_occupancy_mapper.cpp -- and against real depth frames on
// hardware, producing sensible voxel counts); (b) repeated device
// crash/reconnect churn while depthai's own queues are being actively
// read has been observed to eventually corrupt depthai's host-side state
// badly enough to abort the whole process (glibc "corrupted double-
// linked list") -- a robustness gap in the vendored depthai library
// reacting to a flaky device, not a memory-safety bug in this class or
// its callers. Requesting StereoDepth does add real device-side compute
// and USB bandwidth on top of an already-marginal connection, so it's
// plausible (not confirmed) that enabling mapping makes an already-flaky
// session crash more often, even though it isn't the root cause.
// Revisit if/when the underlying device connection is made more robust;
// not something more code here can fix on its own.

#pragma once

#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <memory>
#include <thread>

#include <tbb/concurrent_queue.h>

#include <opencv2/core/mat.hpp>

// The lightweight key/hash-set definitions only -- not the full
// <octomap/octomap.h> (OcTree, Pointcloud, the actual mapping logic),
// which stays a private implementation detail fully contained in
// occupancy_mapper.cpp (tree_ below is a private unique_ptr to an
// otherwise-forward-declared type). Same reasoning as why
// dashboard_client.h doesn't include <nlohmann/json.hpp> even though its
// .cpp uses it throughout -- consumers of this header shouldn't need to
// see octomap's own includes just to hold an OccupancyMapper::Ptr.
#include <octomap/OcTreeKey.h>

#include <basalt/utils/eigen_utils.hpp>

#include <sophus/se3.hpp>

namespace octomap {
class OcTree;
}

namespace basalt {

// One depth frame, already paired with the pose it should be mapped
// against. Built by the caller -- see class comment above.
struct DepthFrameInput {
  using Ptr = std::shared_ptr<DepthFrameInput>;

  int64_t t_ns;
  Sophus::SE3d T_w_c;  // camera pose in world frame at capture time
  // RAW (not loop-closure-corrected) body/IMU pose at capture time --
  // needed alongside T_w_c so a later rebuild() can ask for a FRESH
  // corrected pose at this same t_ns once the pose graph has moved on,
  // instead of being stuck with whatever correction (if any) was live at
  // insertion time. See rebuild()'s own comment for why T_w_c alone,
  // frozen at insertion, can't be retroactively fixed.
  Sophus::SE3d T_w_i_raw;
  cv::Mat depth_mm;    // CV_16UC1, millimeters, 0 == invalid/no return

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// Plain pinhole intrinsics for whatever camera StereoDepth's depth output
// is rectified/aligned to -- see OakDDevice::DepthIntrinsics's comment for
// why this must come from the device's own factory calibration, not
// Basalt's own (of the RAW, un-rectified lens): StereoDepth rectifies
// internally using its own calibration, and a mismatch here produces a
// severely warped point cloud (live-diagnosed on OAK-D Pro W, 2026-09-29
// -- see this file's own header comment). No distortion coefficients,
// deliberately: the rectified depth output is undistorted by construction.
// NOTE: these intrinsics alone aren't sufficient -- the depth frame they're
// applied to must also actually be aligned to the same camera (left/CAM_B)
// they were queried for. See OakDDevice::start()'s setDepthAlign() call and
// its comment for a second, compounding bug (wrong depth alignment socket,
// defaulting to the right camera) found alongside this one, 2026-09-29.
struct DepthIntrinsics {
  double fx = 0, fy = 0, cx = 0, cy = 0;
  int width = 0, height = 0;
};

// One incremental update since the previous insertion, read straight from
// octomap's own change-detection rather than us hand-rolling a diff --
// this is exactly the add/remove batch shape the dashboard transport
// wants (see VoxelBatch in the VIO_Dashboard schema proposal).
struct VoxelDelta {
  Eigen::aligned_vector<Eigen::Vector3d> added;
  Eigen::aligned_vector<Eigen::Vector3d> removed;
};

class OccupancyMapper {
 public:
  using Ptr = std::shared_ptr<OccupancyMapper>;

  // depth_stride: only every Nth pixel (in both u and v) is inserted --
  // occupancy mapping doesn't need every pixel to look right (this is the
  // "semi-dense" idea from the research: budget compute by selecting a
  // subset of pixels, not by using a different depth algorithm), and at
  // depth-camera resolutions a stride of 4 already means the far majority
  // of pixels are still represented at typical voxel sizes.
  //
  // min/max_depth_mm: reject any pixel outside [min, max] before
  // unprojecting it. 0 means "no limit" (the default, kept for
  // test_occupancy_mapper.cpp's synthetic depths, which have nothing to do
  // with real sensor noise). Real hardware needs this: live-diagnosed on
  // OAK-D Pro W, 2026-09-29 -- active-IR stereo pointed at a close, flat,
  // low-texture wall produces genuine high-confidence FALSE matches (the
  // periodic IR dot pattern aliases onto the wrong dot), which read as a
  // coherent phantom surface many meters away rather than random noise, and
  // sail straight through setConfidenceThreshold() since they're
  // internally consistent matches, just to the wrong dot. Hand-verified
  // against real pixels that the *unprojection math* was already correct
  // (see occupancy_mapper.cpp's git history) -- this filter is the actual
  // fix for that symptom, not a coordinate bug.
  OccupancyMapper(const DepthIntrinsics& depth_intrinsics, double voxel_size,
                   int depth_stride = 4, int min_depth_mm = 0,
                   int max_depth_mm = 0);
  ~OccupancyMapper();

  OccupancyMapper(const OccupancyMapper&) = delete;
  OccupancyMapper& operator=(const OccupancyMapper&) = delete;

  void start();
  void stop();

  // Non-blocking, best-effort -- see threading model above.
  void addDepthFrame(const DepthFrameInput::Ptr& frame);

  // Non-blocking poll for the next available incremental update. Returns
  // false (and leaves delta_out untouched) if nothing new has been
  // produced since the last call.
  bool pollVoxelDelta(VoxelDelta& delta_out);

  // Looks up the best CURRENTLY-known corrected camera pose for a frame
  // retained at capture time (t_ns + its original raw body pose) --
  // returns false if no correction is available yet (e.g. no keyframes
  // processed yet), in which case rebuild() skips that frame rather than
  // guessing. Set once, before start(), from oak_d_vio.cpp (wraps
  // OnlineLoopClosure::getCorrectedPoseForRebuild() + the IMU-to-camera
  // extrinsic) -- kept as an injected function rather than a hard
  // dependency so this class still doesn't need to know OnlineLoopClosure
  // or calibration details exist, matching its existing design (see the
  // class header comment: "this class only ever consumes a pose").
  using PoseLookupFn = std::function<bool(
      int64_t t_ns, const Sophus::SE3d& raw_pose_at_capture,
      Sophus::SE3d& corrected_camera_pose_out)>;
  void setPoseLookup(PoseLookupFn fn);

  // Asks the processing thread to rebuild the WHOLE map from retained
  // frames, each re-projected using pose_lookup_'s CURRENT answer instead
  // of whatever pose was live when it was first inserted. See rebuild()'s
  // own comment for why this exists: a live, incrementally-built octree
  // has no way to retroactively fix voxels placed before a later loop-
  // closure correction arrives, so they sit there as permanent ghost
  // layers. Cheap and safe to call often -- internally rate-limited (see
  // kRebuildCooldownS in the .cpp), so e.g. calling this on every single
  // loop-closure event is fine; excess requests are just dropped, not
  // queued up. No-op if setPoseLookup() was never called. Non-blocking.
  void requestRebuild();

 private:
  void processingThreadMain();
  void insertFrame(const DepthFrameInput::Ptr& frame);
  void rebuild();

  // Shared by insertFrame() (one frame's worth of points at a time) and
  // rebuild() (all retained frames' points, in one pass) -- inserts a
  // world-frame point cloud from one pose's origin. Does NOT publish a
  // VoxelDelta itself; call publishChanges() afterward (once per
  // insertFrame() call, but only ONCE total after rebuild()'s whole
  // batch -- see its own comment for why per-frame diffing during a
  // rebuild would be wrong, not just wasteful).
  void insertPointsIntoTree(const Eigen::aligned_vector<Eigen::Vector3d>& points_world,
                             const Eigen::Vector3d& origin);
  // Diffs octomap's own change-detection against occupied_keys_ and
  // pushes the result as one VoxelDelta -- the tail end both
  // insertFrame() and rebuild() share. See occupied_keys_'s comment for
  // why this re-querying approach is correct regardless of how the tree
  // got here.
  void publishChanges();

  DepthIntrinsics depth_intrinsics_;
  int depth_stride_;
  int min_depth_mm_;
  int max_depth_mm_;

  std::unique_ptr<octomap::OcTree> tree_;

  // Which voxel keys we've most recently reported as occupied -- lets us
  // report only real occupied<->not-occupied transitions in VoxelDelta,
  // not every cell octomap's own change-detection touches (which also
  // includes plain unknown-to-known-free cells along every ray, of no
  // interest to a renderer that never drew them in the first place).
  octomap::KeySet occupied_keys_;

  // One entry per frame actually inserted into the tree, kept around so
  // rebuild() can replay them all against fresher pose corrections.
  // Stores only the already-filtered, already-stride-reduced CAMERA-
  // frame points (not the raw depth image -- a full-resolution depth
  // image per frame would blow memory on a long session; this is already
  // exactly what insertFrame() would unproject anyway, just captured once
  // and reused instead of recomputed). Bounded by kMaxRetainedFrames (see
  // the .cpp) -- oldest frames drop off the front once exceeded, same
  // reasoning as any other bounded live buffer in this codebase: a
  // multi-hour session shouldn't grow this without limit.
  struct RetainedFrame {
    int64_t t_ns;
    Sophus::SE3d T_w_i_raw;
    Eigen::aligned_vector<Eigen::Vector3d> points_cam;
  };
  std::deque<RetainedFrame> retained_frames_;

  PoseLookupFn pose_lookup_;
  std::atomic<bool> rebuild_requested_{false};
  // Wall-clock time of the last actual rebuild -- see kRebuildCooldownS.
  // Not initialized to "now" so the very first request isn't throttled.
  std::chrono::steady_clock::time_point last_rebuild_wall_{};
  bool has_rebuilt_once_ = false;

  std::atomic<bool> running_{false};
  std::thread thread_;

  tbb::concurrent_bounded_queue<DepthFrameInput::Ptr> input_queue_;
  tbb::concurrent_bounded_queue<VoxelDelta> output_queue_;
};

}  // namespace basalt
