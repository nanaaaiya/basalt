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

#include <octomap/octomap.h>

namespace basalt {

OccupancyMapper::OccupancyMapper(const Calibration<double>& calib,
                                  double voxel_size, int depth_stride)
    : calib_(calib), depth_stride_(std::max(1, depth_stride)) {
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

void OccupancyMapper::processingThreadMain() {
  while (true) {
    DepthFrameInput::Ptr frame;
    input_queue_.pop(frame);
    if (!frame) break;  // shutdown sentinel
    insertFrame(frame);
  }
}

void OccupancyMapper::insertFrame(const DepthFrameInput::Ptr& frame) {
  if (frame->cam_id < 0 ||
      static_cast<size_t>(frame->cam_id) >= calib_.intrinsics.size()) {
    return;
  }
  const auto& cam = calib_.intrinsics[frame->cam_id];
  const cv::Mat& depth = frame->depth_mm;

  // The depth node is free to output at a lower resolution than the
  // calibration's (DepthAI's StereoDepth default preset does this --
  // observed 320x240 depth frames against a 640x480 calibration). The
  // intrinsics (fx, fy, cx, cy) are only valid in the calibration's own
  // resolution, so a raw depth-image pixel must be rescaled into that
  // resolution before unprojecting -- otherwise every pixel is
  // back-projected through the wrong focal length/principal point,
  // distorting the point cloud more the further a pixel is from center.
  // This keeps the fix correct at whatever resolution the depth stream
  // actually outputs, rather than forcing the device to output at full
  // calibration resolution (which costs real USB bandwidth and can
  // destabilize an already-marginal connection).
  double scale_u = 1.0, scale_v = 1.0;
  if (static_cast<size_t>(frame->cam_id) < calib_.resolution.size() &&
      depth.cols > 0 && depth.rows > 0) {
    const Eigen::Vector2i& calib_res = calib_.resolution[frame->cam_id];
    scale_u = static_cast<double>(calib_res.x()) / depth.cols;
    scale_v = static_cast<double>(calib_res.y()) / depth.rows;
  }

  // Back-project every depth_stride_'th pixel into a world-frame point
  // cloud. Each pixel's camera model unproject() gives a unit bearing
  // vector (not a z=1-plane point -- see basalt-headers' camera models),
  // so it's rescaled along that ray until its z-component equals the
  // depth value: standard "depth image" semantics (perpendicular distance
  // from the image plane), matching what DepthAI's StereoDepth outputs.
  octomap::Pointcloud cloud;
  for (int v = 0; v < depth.rows; v += depth_stride_) {
    const uint16_t* row = depth.ptr<uint16_t>(v);
    for (int u = 0; u < depth.cols; u += depth_stride_) {
      uint16_t d_mm = row[u];
      if (d_mm == 0) continue;  // no valid return at this pixel

      Eigen::Vector2d proj(static_cast<double>(u) * scale_u,
                            static_cast<double>(v) * scale_v);
      Eigen::Vector3d bearing;
      if (!cam.unproject(proj, bearing)) continue;
      if (bearing.z() <= 1e-6) continue;  // behind or parallel to the image plane

      double z_m = d_mm / 1000.0;
      Eigen::Vector3d p_cam = bearing * (z_m / bearing.z());
      Eigen::Vector3d p_world = frame->T_w_c * p_cam;
      cloud.push_back(static_cast<float>(p_world.x()),
                       static_cast<float>(p_world.y()),
                       static_cast<float>(p_world.z()));
    }
  }

  if (cloud.size() == 0) return;

  const Eigen::Vector3d& origin = frame->T_w_c.translation();
  tree_->insertPointCloud(
      cloud, octomap::point3d(static_cast<float>(origin.x()),
                               static_cast<float>(origin.y()),
                               static_cast<float>(origin.z())));

  // octomap's own change-detection gives us the changed-cell set since
  // the last reset, but that includes every plain unknown-to-known-free
  // cell along each ray too -- of no interest to a renderer that never
  // drew them. Only report a cell when its occupied/not-occupied state,
  // as we've most recently told the caller about it, actually flips (see
  // occupied_keys_'s comment in the header). Re-querying each key's
  // current occupancy (rather than trusting whatever the change-detection
  // map's own stored value means) is deliberate: it's correct regardless
  // of that map's exact internal semantics.
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

}  // namespace basalt
