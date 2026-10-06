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

#include <basalt/mapping/occupancy_mapper.h>

#include <algorithm>
#include <iostream>

#include <basalt/utils/thread_priority.h>

#include <octomap/octomap.h>

namespace basalt {

OccupancyMapper::OccupancyMapper(const DepthIntrinsics& depth_intrinsics,
                                  double voxel_size, int depth_stride,
                                  int min_depth_mm, int max_depth_mm)
    : depth_intrinsics_(depth_intrinsics),
      depth_stride_(std::max(1, depth_stride)),
      min_depth_mm_(min_depth_mm),
      max_depth_mm_(max_depth_mm) {
  tree_.reset(new octomap::OcTree(voxel_size));
  tree_->enableChangeDetection(true);
  input_queue_.set_capacity(4);
  output_queue_.set_capacity(200);
}

OccupancyMapper::~OccupancyMapper() { stop(); }

void OccupancyMapper::start() {
  if (running_.exchange(true)) return;
  thread_ = std::thread([this] { processingThreadMain(); });
}

void OccupancyMapper::stop() {
  if (!running_.exchange(false)) return;
  // Blocking push, matching MargDataFanOut/DashboardClient's own shutdown
  // idiom: guarantees the processing thread's blocking pop() wakes up,
  // rather than racing to hope the queue happens to be non-empty already
  // (see the real DashboardClient::stop() deadlock this was modeled to
  // avoid).
  input_queue_.push(nullptr);
  if (thread_.joinable()) thread_.join();
}

void OccupancyMapper::addDepthFrame(const DepthFrameInput::Ptr& frame) {
  input_queue_.try_push(frame);
}

bool OccupancyMapper::pollVoxelDelta(VoxelDelta& delta_out) {
  return output_queue_.try_pop(delta_out);
}

void OccupancyMapper::setPoseLookup(PoseLookupFn fn) {
  pose_lookup_ = std::move(fn);
}

void OccupancyMapper::requestRebuild() { rebuild_requested_.store(true); }

// How often rebuild() is actually allowed to run, no matter how often
// requestRebuild() gets called -- e.g. a live loop-closure-heavy burst can
// fire many times a second (observed directly, 2026-09-29/30 sessions), and
// a full rebuild replays every retained frame, not just one. Excess
// requests inside the cooldown are simply dropped (see requestRebuild()'s
// own comment) -- the NEXT request after the cooldown expires picks it back
// up, so nothing needs to be queued. A reasoned starting point, not a
// live-tuned value -- same caveat as this file's other constants.
constexpr double kRebuildCooldownS = 3.0;
// Caps retained_frames_ -- see its own header comment. At the default
// occupancy_rate_hz=5, 3000 frames is ~10 minutes of a live session.
constexpr size_t kMaxRetainedFrames = 3000;

void OccupancyMapper::processingThreadMain() {
  lowerCurrentThreadPriority();
  while (true) {
    DepthFrameInput::Ptr frame;
    if (input_queue_.try_pop(frame)) {
      if (!frame) break;  // shutdown sentinel
      insertFrame(frame);
      continue;  // prioritize draining new frames over a pending rebuild
    }
    if (rebuild_requested_.exchange(false)) {
      rebuild();
      continue;
    }
    // Neither queue had anything ready -- avoid a busy-spin. Short enough
    // that a shutdown (stop() flips running_ then pushes the sentinel)
    // still feels immediate, long enough not to burn a core doing nothing.
    if (!running_.load()) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}

void OccupancyMapper::insertFrame(const DepthFrameInput::Ptr& frame) {
  const cv::Mat& depth = frame->depth_mm;
  if (depth_intrinsics_.width <= 0 || depth_intrinsics_.height <= 0) return;

  // The depth node is free to output at a lower resolution than
  // depth_intrinsics_'s own reference resolution (DepthAI's StereoDepth
  // default preset does this -- observed 320x240 depth frames against a
  // 640x480 query). fx/fy/cx/cy are only valid at that reference
  // resolution, so a raw depth-image pixel must be rescaled into it
  // before unprojecting -- otherwise every pixel is back-projected
  // through the wrong focal length/principal point, distorting the point
  // cloud more the further a pixel is from center. This keeps the fix
  // correct at whatever resolution the depth stream actually outputs,
  // rather than forcing the device to output at full resolution (which
  // costs real USB bandwidth and can destabilize an already-marginal
  // connection).
  double scale_u = 1.0, scale_v = 1.0;
  if (depth.cols > 0 && depth.rows > 0) {
    scale_u = static_cast<double>(depth_intrinsics_.width) / depth.cols;
    scale_v = static_cast<double>(depth_intrinsics_.height) / depth.rows;
  }

  // Back-project every depth_stride_'th pixel into a world-frame point
  // cloud, using PLAIN pinhole unprojection (no distortion model) --
  // see DepthIntrinsics's header comment for why: StereoDepth's own
  // output is already rectified/undistorted by construction, and using
  // Basalt's own (RAW-lens) camera model here previously produced a
  // severely warped point cloud on OAK-D Pro W, live-diagnosed
  // 2026-09-29 (a 21% focal-length mismatch plus a completely different
  // distortion model family vs. Luxonis's own factory calibration, which
  // is what StereoDepth actually rectifies against). Each pixel's ray is
  // rescaled along its bearing until its z-component equals the depth
  // value: standard "depth image" semantics (perpendicular distance from
  // the image plane), matching what DepthAI's StereoDepth outputs.
  Eigen::aligned_vector<Eigen::Vector3d> points_cam;
  for (int v = 0; v < depth.rows; v += depth_stride_) {
    const uint16_t* row = depth.ptr<uint16_t>(v);
    for (int u = 0; u < depth.cols; u += depth_stride_) {
      uint16_t d_mm = row[u];
      if (d_mm == 0) continue;  // no valid return at this pixel
      // See DepthIntrinsics's header comment / this class's constructor
      // comment: rejects high-confidence but wrong stereo matches (a real,
      // observed failure mode of active-IR depth on close, flat,
      // low-texture surfaces), not just a sanity clamp. min_depth_mm_ is
      // checked on the raw (forward) depth value -- fine, near-field
      // clipping doesn't meaningfully depend on viewing angle. max_depth_mm_
      // is deliberately NOT checked here: see below, it has to be checked
      // on the actual 3D point, not this raw value.
      if (min_depth_mm_ > 0 && d_mm < min_depth_mm_) continue;

      double u_calib = static_cast<double>(u) * scale_u;
      double v_calib = static_cast<double>(v) * scale_v;
      double bx = (u_calib - depth_intrinsics_.cx) / depth_intrinsics_.fx;
      double by = (v_calib - depth_intrinsics_.cy) / depth_intrinsics_.fy;

      double z_m = d_mm / 1000.0;
      Eigen::Vector3d p_cam(bx * z_m, by * z_m, z_m);
      // max_depth_mm_ has to cap the true 3D distance (p_cam.norm()), not
      // the raw depth-image value d_mm: on a wide-FOV lens, an off-axis
      // pixel's bearing ray points well away from straight-ahead, so its
      // actual distance from the camera can be far more than its forward
      // (z) depth value. Checking d_mm alone let edge-of-frame pixels at
      // exactly the cutoff sail through at much greater real distance --
      // live-diagnosed on OAK-D Pro W, 2026-09-29: a near-stationary
      // camera (pose within 0.2m of origin the whole run) still produced
      // voxels up to 5.2m away, matching this camera's real intrinsics'
      // worst-case corner ratio (z * sqrt(1+bx^2+by^2) ~= z * 1.74) at
      // the old 3000mm cutoff almost exactly (3.0 * 1.74 ~= 5.2m).
      if (max_depth_mm_ > 0 && p_cam.norm() * 1000.0 > max_depth_mm_) {
        continue;
      }
      points_cam.push_back(p_cam);
    }
  }

  if (points_cam.empty()) return;

  Eigen::aligned_vector<Eigen::Vector3d> points_world;
  points_world.reserve(points_cam.size());
  for (const auto& p_cam : points_cam) points_world.push_back(frame->T_w_c * p_cam);
  insertPointsIntoTree(points_world, frame->T_w_c.translation());
  publishChanges();

  // Retained AFTER insertion, with the points already computed above --
  // see RetainedFrame's header comment for why this is cheap (reusing
  // work already done, not a second unprojection pass).
  // Only worth the memory (~1 MB/frame at stride 2) if a rebuild can
  // actually use it.
  if (!pose_lookup_) return;
  retained_frames_.push_back(
      RetainedFrame{frame->t_ns, frame->T_w_i_raw, std::move(points_cam)});
  while (retained_frames_.size() > kMaxRetainedFrames) {
    retained_frames_.pop_front();
  }
}

void OccupancyMapper::insertPointsIntoTree(
    const Eigen::aligned_vector<Eigen::Vector3d>& points_world,
    const Eigen::Vector3d& origin) {
  if (points_world.empty()) return;
  octomap::Pointcloud cloud;
  for (const auto& p : points_world) {
    cloud.push_back(static_cast<float>(p.x()), static_cast<float>(p.y()),
                     static_cast<float>(p.z()));
  }
  tree_->insertPointCloud(
      cloud, octomap::point3d(static_cast<float>(origin.x()),
                               static_cast<float>(origin.y()),
                               static_cast<float>(origin.z())));
}

void OccupancyMapper::publishChanges() {
  // octomap's own change-detection gives us the changed-cell set since
  // the last reset, but that includes every plain unknown-to-known-free
  // cell along each ray too -- of no interest to a renderer that never
  // drew them. Only report a cell when its occupied/not-occupied state,
  // as we've most recently told the caller about it, actually flips (see
  // occupied_keys_'s comment in the header). Re-querying each key's
  // current occupancy (rather than trusting whatever the change-detection
  // map's own stored value means) is deliberate: it's correct regardless
  // of that map's exact internal semantics -- including after rebuild()'s
  // clear()+reinsert-everything, where it's what makes the published
  // delta the correct NET change (only cells that actually ended up in a
  // different state than before the rebuild), not a spurious remove-then-
  // readd of every voxel that happened to survive unchanged.
  VoxelDelta delta;
  for (auto it = tree_->changedKeysBegin(); it != tree_->changedKeysEnd();
       ++it) {
    const octomap::OcTreeKey& key = it->first;
    octomap::OcTreeNode* node = tree_->search(key);
    bool now_occupied = node != nullptr && tree_->isNodeOccupied(node);
    bool was_occupied = occupied_keys_.count(key) > 0;
    if (now_occupied == was_occupied) continue;  // nothing changed for us

    octomap::point3d center = tree_->keyToCoord(key);
    Eigen::Vector3d p(center.x(), center.y(), center.z());
    if (now_occupied) {
      delta.added.push_back(p);
      occupied_keys_.insert(key);
    } else {
      delta.removed.push_back(p);
      occupied_keys_.erase(key);
    }
  }
  tree_->resetChangeDetection();

  if (!delta.added.empty() || !delta.removed.empty()) {
    output_queue_.try_push(std::move(delta));
  }
}

void OccupancyMapper::rebuild() {
  // See this class's header comment on requestRebuild()/RetainedFrame for
  // why this exists: a live, incrementally-built octree has no way to
  // retroactively fix voxels placed using a pose that a LATER loop-closure
  // correction has since revised -- they just sit there as permanent
  // ghost layers (exactly the "map got created multiple times and
  // overlaps on itself" symptom live-diagnosed 2026-10-01). This clears
  // the whole tree and replays every retained frame using pose_lookup_'s
  // CURRENT best answer instead of the pose that was live at insertion
  // time, then publishes ONE net diff -- so a voxel that didn't actually
  // move between rebuilds produces no spurious add/remove churn on the
  // dashboard, only real changes do.
  // TEMPORARY diagnostic logging -- added 2026-10-01 after a live test
  // showed no visible improvement and there was no way to tell from the
  // console whether rebuild() ran at all, skipped, or ran but didn't
  // change much. Remove once this path is confirmed working live.
  if (!pose_lookup_) {
    std::cerr << "[OCCMAP-REBUILD] skipped: no pose_lookup_ set"
              << std::endl;
    return;  // no caller wired one up -- nothing to do
  }

  auto now = std::chrono::steady_clock::now();
  if (has_rebuilt_once_) {
    double elapsed =
        std::chrono::duration<double>(now - last_rebuild_wall_).count();
    if (elapsed < kRebuildCooldownS) {
      std::cerr << "[OCCMAP-REBUILD] skipped: cooldown (" << elapsed << "s < "
                << kRebuildCooldownS << "s)" << std::endl;
      return;  // see kRebuildCooldownS
    }
  }

  tree_->clear();
  tree_->enableChangeDetection(true);  // defensive -- see constructor

  size_t used = 0, skipped_no_pose = 0;
  for (const auto& rf : retained_frames_) {
    Sophus::SE3d T_w_c;
    if (!pose_lookup_(rf.t_ns, rf.T_w_i_raw, T_w_c)) {
      ++skipped_no_pose;
      continue;
    }
    ++used;
    Eigen::aligned_vector<Eigen::Vector3d> points_world;
    points_world.reserve(rf.points_cam.size());
    for (const auto& p_cam : rf.points_cam) points_world.push_back(T_w_c * p_cam);
    insertPointsIntoTree(points_world, T_w_c.translation());
  }

  size_t occupied_before = occupied_keys_.size();
  publishChanges();
  last_rebuild_wall_ = now;
  has_rebuilt_once_ = true;
  std::cerr << "[OCCMAP-REBUILD] retained_frames=" << retained_frames_.size()
            << " used=" << used << " skipped_no_pose=" << skipped_no_pose
            << " occupied_before=" << occupied_before
            << " occupied_after=" << occupied_keys_.size() << std::endl;
}

}  // namespace basalt
