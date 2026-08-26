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

// Online (live) loop-closure module, inspired by VINS-Fusion's loop_fusion
// node: a separate consumer of the VIO front-end's keyframe stream that
// detects revisits via bag-of-words place recognition and maintains an
// incrementally-corrected 4-DOF (x, y, z, yaw) pose graph, publishing a
// continuously-updated globally-consistent trajectory alongside the raw VIO
// estimate. Unlike VINS-Fusion, this reuses Basalt's own already-built
// keypoint detection / BoW indexing / RANSAC verification primitives (see
// nfr_mapper.cpp, hash_bow.h) rather than DBoW2 + Ceres -- those primitives
// are already incremental (HashBow::add_to_database/querry_database operate
// one keyframe at a time), so this module is mostly new glue plus one
// genuinely new piece: a small, fast pose-graph optimizer, deliberately
// pose-only (no landmarks) so it stays cheap regardless of session length,
// unlike Basalt's own NfrMapper::optimize() (full nonlinear-factor-recovery
// bundle adjustment, meant for offline batch use in basalt_mapper).
//
// Loop-edge relative poses are recovered via stereo-triangulated 3D points
// (metric scale, from the loop-candidate keyframe's own calibrated stereo
// pair) + PnP-RANSAC of the new keyframe's 2D keypoints against them --
// deliberately NOT plain monocular 2D-2D matching, which only recovers
// translation *direction*, not metric magnitude, and would silently feed
// wrongly-scaled corrections into the pose graph.

#pragma once

#include <atomic>
#include <mutex>
#include <thread>
#include <unordered_map>

#include <tbb/concurrent_queue.h>

#include <basalt/calibration/calibration.hpp>
#include <basalt/hash_bow/hash_bow.h>
#include <basalt/utils/common_types.h>
#include <basalt/utils/imu_types.h>
#include <basalt/utils/vio_config.h>

namespace basalt {

// Result of a single verified visual match against a historical keyframe --
// published the moment PnP-RANSAC succeeds, before pose-graph insertion or
// solving happens. Answers "given this known reference keyframe, where am I
// NOW?" for consumers (navigation, RTL) that need the freshest possible
// localization and shouldn't have to wait for a global pose-graph solve.
struct LocalizationResult {
  int64_t t_ns = 0;                  // current keyframe's timestamp
  int64_t reference_t_ns = 0;        // matched historical keyframe's timestamp
  Sophus::SE3d T_reference_current;  // reference body frame -> current body
                                      // frame (the raw relative-pose measurement)
  Sophus::SE3d T_w_current;          // world frame -> current body frame,
                                      // composed with the reference keyframe's
                                      // current CORRECTED pose (not raw), so
                                      // this already reflects any earlier
                                      // pose-graph corrections without waiting
                                      // for a fresh solve
  int num_inliers = 0;
};

class OnlineLoopClosure {
 public:
  using Ptr = std::shared_ptr<OnlineLoopClosure>;

  OnlineLoopClosure(const Calibration<double>& calib, const VioConfig& config);
  ~OnlineLoopClosure();

  // Feed this from vio->out_marg_queue.
  tbb::concurrent_bounded_queue<MargData::Ptr> input_queue;

  // Published immediately on every successful verified match -- see
  // LocalizationResult above. Non-blocking on the producer side (try_push),
  // so a slow/absent consumer never stalls keyframe processing.
  tbb::concurrent_bounded_queue<LocalizationResult> localization_queue;

  void start();
  void stop();

  // Thread-safe snapshot of the latest globally-corrected trajectory,
  // ordered by keyframe arrival (== time order).
  Eigen::aligned_vector<Eigen::Vector3d> getCorrectedTrajectory() const;
  bool getLatestCorrectedPose(Sophus::SE3d& out) const;

  // Same as getCorrectedTrajectory(), but paired with each keyframe's
  // timestamp -- needed for logging/analysis (matching timestamps up
  // against the raw VIO trajectory, sample rate, etc.), not just drawing a
  // line in the GUI.
  void getCorrectedTrajectoryWithTimestamps(
      std::vector<int64_t>& t_ns,
      Eigen::aligned_vector<Eigen::Vector3d>& positions) const;

  int numLoopClosures() const { return num_loop_closures.load(); }

 private:
  struct LoopKeyframe {
    int64_t t_ns;

    KeypointsData kd0;  // cam0 2D keypoints/descriptors/bow for this keyframe

    // Metric 3D points (in this keyframe's own cam0 frame) from stereo
    // triangulation, used as the PnP target when a LATER keyframe matches
    // against this one.
    Eigen::aligned_vector<Eigen::Vector3d> pts3d;
    std::unordered_map<int, int> corner_to_pt3d;  // kd0.corners idx -> pts3d idx

    // Raw (uncorrected) VIO pose at this keyframe -- sequential-edge
    // measurements always come from here, never from the corrected state
    // below, to avoid feeding corrections back into their own inputs.
    Sophus::SE3d T_w_i_raw;

    // Pose graph state: roll/pitch fixed (from raw VIO pose, never
    // corrected -- IMU/gravity already observes these well); yaw + t are
    // the solved-for unknowns.
    double roll = 0, pitch = 0, yaw = 0;
    Eigen::Vector3d t_opt = Eigen::Vector3d::Zero();
  };

  struct PoseGraphEdge {
    size_t i, j;
    Eigen::Vector3d dt;
    double dyaw;
    // Relative confidence of this edge's measurement, applied as a scalar
    // weight on its residual in the solve (equivalent to scaling its
    // information matrix). Odometry edges keep the default 1.0; loop edges
    // get inlier_count / mapper_min_matches (clamped), so a well-supported
    // closure pulls the graph harder than one that barely cleared the
    // acceptance floor -- previously every edge was weighted identically
    // regardless of how many inliers actually backed it.
    double weight = 1.0;
  };

  void processingLoop();
  void processKeyframe(const MargData::Ptr& data, int64_t kf_id);
  void solvePoseGraph();

  Calibration<double> calib_;
  VioConfig config_;

  std::unique_ptr<HashBow<256>> hash_bow_;

  std::vector<LoopKeyframe> keyframes_;  // arrival order == time order
  std::unordered_map<int64_t, size_t> t_ns_to_idx_;
  std::vector<PoseGraphEdge> edges_;

  // Keyframe-storage distance gate state (see
  // config_.mapper_min_keyframe_storage_dist): position (raw VIO estimate)
  // of the last keyframe actually added to the match database, so the next
  // candidate's distance can be measured against it rather than against
  // every keyframe Basalt hands us (most of which never get stored at all).
  Eigen::Vector3d last_stored_raw_position_ = Eigen::Vector3d::Zero();
  bool has_stored_any_ = false;

  mutable std::mutex state_mutex_;
  std::atomic<int> num_loop_closures{0};

  std::atomic<bool> running{false};
  std::thread worker_thread_;
};

}  // namespace basalt
