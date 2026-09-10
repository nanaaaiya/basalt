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

#include <basalt/calibration/calibration.hpp>
#include <basalt/utils/eigen_utils.hpp>

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
  cv::Mat depth_mm;    // CV_16UC1, millimeters, 0 == invalid/no return
  int cam_id = 0;      // which calib_.intrinsics[cam_id] applies

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
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
  OccupancyMapper(const Calibration<double>& calib, double voxel_size,
                   int depth_stride = 4);
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

 private:
  void processingThreadMain();
  void insertFrame(const DepthFrameInput::Ptr& frame);

  Calibration<double> calib_;
  int depth_stride_;

  std::unique_ptr<octomap::OcTree> tree_;

  // Which voxel keys we've most recently reported as occupied -- lets us
  // report only real occupied<->not-occupied transitions in VoxelDelta,
  // not every cell octomap's own change-detection touches (which also
  // includes plain unknown-to-known-free cells along every ray, of no
  // interest to a renderer that never drew them in the first place).
  octomap::KeySet occupied_keys_;

  std::atomic<bool> running_{false};
  std::thread thread_;

  tbb::concurrent_bounded_queue<DepthFrameInput::Ptr> input_queue_;
  tbb::concurrent_bounded_queue<VoxelDelta> output_queue_;
};

}  // namespace basalt
