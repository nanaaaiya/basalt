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

#include <basalt/vi_estimator/online_loop_closure.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

#include <basalt/utils/keypoints.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#pragma GCC diagnostic pop

namespace basalt {

namespace {

// Decompose R = Rz(yaw) * Ry(pitch) * Rx(roll). Assumes no gimbal lock
// (pitch away from +-90 deg), a reasonable assumption for a handheld/mobile
// device that isn't doing full vertical flips.
void decomposeYPR(const Eigen::Matrix3d& R, double& roll, double& pitch,
                  double& yaw) {
  pitch = std::asin(std::clamp(-R(2, 0), -1.0, 1.0));
  roll = std::atan2(R(2, 1), R(2, 2));
  yaw = std::atan2(R(1, 0), R(0, 0));
}

Eigen::Matrix3d composeYPR(double roll, double pitch, double yaw) {
  return (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
          Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()))
      .toRotationMatrix();
}

double wrapAngle(double a) {
  while (a > M_PI) a -= 2 * M_PI;
  while (a < -M_PI) a += 2 * M_PI;
  return a;
}

// Mirrors matchFastHelper's nearest/second-nearest search (see
// src/utils/keypoints.cpp) one direction only, but reports the two
// intermediate counts matchDescriptors() itself doesn't expose: how many
// descriptors in d1 have a nearest neighbor within the raw Hamming
// threshold at all (after_hamming), vs. how many of those also pass the
// second-best-match ratio test (after_ratio) -- purely for diagnostic
// logging, doesn't affect the actual matches used downstream (those still
// come from the real matchDescriptors(), including its mutual cross-check).
void countMatchStages(const std::vector<std::bitset<256>>& d1,
                      const std::vector<std::bitset<256>>& d2, int threshold,
                      double ratio, int& after_hamming, int& after_ratio) {
  after_hamming = 0;
  after_ratio = 0;
  for (size_t i = 0; i < d1.size(); i++) {
    int best_dist = 500, best2_dist = 500;
    for (size_t j = 0; j < d2.size(); j++) {
      int dist = (int)(d1[i] ^ d2[j]).count();
      if (dist <= best_dist) {
        best2_dist = best_dist;
        best_dist = dist;
      } else if (dist < best2_dist) {
        best2_dist = dist;
      }
    }
    if (best_dist < threshold) {
      after_hamming++;
      if (best_dist * ratio <= best2_dist) after_ratio++;
    }
  }
}

// Midpoint-of-closest-approach triangulation of two rays given in a common
// frame: ray0 from origin O0=0 with direction d0, ray1 from origin O1 with
// direction d1 (both normalized). Returns false if the rays are near-
// parallel or the intersection lies behind either camera.
bool triangulateMidpoint(const Eigen::Vector3d& d0, const Eigen::Vector3d& O1,
                         const Eigen::Vector3d& d1, Eigen::Vector3d& point) {
  double b = d0.dot(d1);
  double denom = 1.0 - b * b;
  if (std::abs(denom) < 1e-9) return false;

  Eigen::Vector3d w0 = -O1;
  double dd = d0.dot(w0);
  double e = d1.dot(w0);
  double s = (b * e - dd) / denom;
  double t = (e - b * dd) / denom;
  if (s <= 0.05 || t <= 0.05) return false;

  Eigen::Vector3d P0 = s * d0;
  Eigen::Vector3d P1 = O1 + t * d1;
  point = 0.5 * (P0 + P1);

  return point.norm() <= 30.0;  // reject absurdly-far triangulations
}

}  // namespace

OnlineLoopClosure::OnlineLoopClosure(const Calibration<double>& calib,
                                     const VioConfig& config)
    : calib_(calib), config_(config) {
  hash_bow_.reset(new HashBow<256>(config_.mapper_bow_num_bits));
  input_queue.set_capacity(1000);
}

OnlineLoopClosure::~OnlineLoopClosure() { stop(); }

void OnlineLoopClosure::start() {
  if (running.exchange(true)) return;
  worker_thread_ = std::thread(&OnlineLoopClosure::processingLoop, this);
}

void OnlineLoopClosure::stop() {
  if (!running.exchange(false)) return;
  input_queue.push(nullptr);
  if (worker_thread_.joinable()) worker_thread_.join();
}

void OnlineLoopClosure::processingLoop() {
  MargData::Ptr data;
  while (true) {
    input_queue.pop(data);
    if (!data.get()) break;
    if (data->kfs_to_marg.empty()) continue;

    int64_t kf_id = *data->kfs_to_marg.begin();
    processKeyframe(data, kf_id);
  }
  std::cout << "Finished OnlineLoopClosure" << std::endl;
}

void OnlineLoopClosure::processKeyframe(const MargData::Ptr& data,
                                        int64_t kf_id) {
  auto pose_it = data->frame_poses.find(kf_id);
  if (pose_it == data->frame_poses.end()) return;
  Sophus::SE3d T_w_i_raw = pose_it->second.getPose();

  OpticalFlowResult::Ptr res;
  for (const auto& r : data->opt_flow_res) {
    if (r.get() && r->t_ns == kf_id) {
      res = r;
      break;
    }
  }
  if (!res.get() || !res->input_images.get() ||
      res->input_images->img_data.size() < 2)
    return;
  if (!res->input_images->img_data[0].img.get() ||
      !res->input_images->img_data[1].img.get())
    return;

  const Image<const uint16_t> img0 =
      res->input_images->img_data[0].img->Reinterpret<const uint16_t>();
  const Image<const uint16_t> img1 =
      res->input_images->img_data[1].img->Reinterpret<const uint16_t>();

  // TIMING DIAGNOSTIC (help isolate which stage actually causes the
  // reported lag, rather than guessing): checkpoints around each candidate
  // cause -- cam0 detection, stereo triangulation, loop-candidate search,
  // and the pose-graph solve. Printed once per processed keyframe.
  auto t0 = std::chrono::steady_clock::now();

  LoopKeyframe kf;
  kf.t_ns = kf_id;
  kf.T_w_i_raw = T_w_i_raw;

  detectKeypointsMapping(img0, kf.kd0, config_.mapper_detection_num_points);
  computeAngles(img0, kf.kd0, true);
  computeDescriptors(img0, kf.kd0);
  {
    std::vector<bool> success;
    calib_.intrinsics[0].unproject(kf.kd0.corners, kf.kd0.corners_3d, success);
  }
  hash_bow_->compute_bow(kf.kd0.corner_descriptors, kf.kd0.hashes,
                         kf.kd0.bow_vector);

  auto t1 = std::chrono::steady_clock::now();

  // Stereo triangulation: metric 3D points in this keyframe's own cam0
  // frame, stored so a LATER keyframe can PnP against them.
  {
    KeypointsData kd1;
    detectKeypointsMapping(img1, kd1, config_.mapper_detection_num_points);
    computeAngles(img1, kd1, true);
    computeDescriptors(img1, kd1);

    std::vector<std::pair<int, int>> matches;
    matchDescriptors(kf.kd0.corner_descriptors, kd1.corner_descriptors, matches,
                     (int)config_.mapper_max_hamming_distance,
                     config_.mapper_second_best_test_ratio);

    Sophus::SE3d T_c0_c1 = calib_.T_i_c[0].inverse() * calib_.T_i_c[1];
    Eigen::Vector3d O1 = T_c0_c1.translation();

    for (const auto& m : matches) {
      Eigen::Vector4d b0h, b1h;
      if (!calib_.intrinsics[0].unproject(kf.kd0.corners[m.first], b0h))
        continue;
      if (!calib_.intrinsics[1].unproject(kd1.corners[m.second], b1h)) continue;

      Eigen::Vector3d d0 = b0h.head<3>().normalized();
      Eigen::Vector3d d1 = (T_c0_c1.so3() * b1h.head<3>()).normalized();

      Eigen::Vector3d point;
      if (!triangulateMidpoint(d0, O1, d1, point)) continue;

      int pt_idx = (int)kf.pts3d.size();
      kf.pts3d.push_back(point);
      kf.corner_to_pt3d[m.first] = pt_idx;
    }
  }

  auto t2 = std::chrono::steady_clock::now();

  // --- Loop detection: query BoW DB restricted to earlier keyframes ---
  bool have_loop = false;
  size_t loop_partner_idx = 0;
  Sophus::SE3d T_body_partner_new;

  {
    std::vector<std::pair<TimeCamId, double>> results;
    hash_bow_->querry_database(kf.kd0.bow_vector,
                               (size_t)config_.mapper_num_frames_to_match,
                               results, &kf_id);

    std::vector<std::pair<TimeCamId, double>> above_threshold;
    for (const auto& cand : results) {
      if (cand.second >= config_.mapper_frames_to_match_threshold)
        above_threshold.push_back(cand);
    }

    size_t this_kf_display_id = keyframes_.size();

    if (above_threshold.empty()) {
      std::cout << "[ONLINE-LOOP] kf=" << this_kf_display_id
                << " (t_ns=" << kf_id << ") candidates=0 (after BoW threshold)\n"
                << "              RESULT: rejected (no candidates above "
                   "bow threshold "
                << config_.mapper_frames_to_match_threshold << ")"
                << std::endl;
    }

    for (const auto& cand : above_threshold) {
      const LoopKeyframe* partner = nullptr;
      size_t partner_idx = 0;
      int64_t partner_t_ns = 0;
      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        auto it = t_ns_to_idx_.find(cand.first.frame_id);
        if (it == t_ns_to_idx_.end()) continue;
        partner_idx = it->second;
        partner = &keyframes_[partner_idx];
        partner_t_ns = partner->t_ns;
        if (partner->pts3d.empty()) continue;
      }

      int after_hamming = 0, after_ratio = 0;
      countMatchStages(kf.kd0.corner_descriptors, partner->kd0.corner_descriptors,
                       (int)config_.mapper_max_hamming_distance,
                       config_.mapper_second_best_test_ratio, after_hamming,
                       after_ratio);

      std::vector<std::pair<int, int>> matches;
      matchDescriptors(kf.kd0.corner_descriptors, partner->kd0.corner_descriptors,
                       matches, (int)config_.mapper_max_hamming_distance,
                       config_.mapper_second_best_test_ratio);

      opengv::bearingVectors_t bearingVectors;
      opengv::points_t points;

      for (const auto& m : matches) {
        auto pit = partner->corner_to_pt3d.find(m.second);
        if (pit == partner->corner_to_pt3d.end()) continue;

        Eigen::Vector4d bh;
        if (!calib_.intrinsics[0].unproject(kf.kd0.corners[m.first], bh))
          continue;

        bearingVectors.push_back(bh.head<3>().normalized());
        points.push_back(partner->pts3d[pit->second]);
      }

      std::cout << "[ONLINE-LOOP] kf=" << this_kf_display_id
                << " (t_ns=" << kf_id << ") candidates=" << above_threshold.size()
                << " (after BoW threshold)\n"
                << "              best_candidate=kf" << partner_idx
                << " (t_ns=" << partner_t_ns << ") bow_score=" << cand.second
                << "\n"
                << "              matches_after_hamming=" << after_hamming
                << "\n"
                << "              matches_after_ratio_test=" << after_ratio
                << " (mutual_cross_check=" << matches.size() << ")\n"
                << "              pnp_ready_points=" << bearingVectors.size()
                << " | partner_triangulated=" << partner->pts3d.size() << "/"
                << partner->kd0.corners.size() << " corners ("
                << (partner->kd0.corners.empty()
                        ? 0.0
                        : 100.0 * partner->pts3d.size() /
                              partner->kd0.corners.size())
                << "%)" << std::endl;

      if ((int)bearingVectors.size() < config_.mapper_min_matches) {
        std::cout << "              RESULT: rejected (pnp-ready matches "
                  << bearingVectors.size() << " < mapper_min_matches "
                  << config_.mapper_min_matches << ")" << std::endl;
        continue;
      }

      opengv::absolute_pose::CentralAbsoluteAdapter adapter(bearingVectors,
                                                             points);
      opengv::sac::Ransac<
          opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
          ransac;
      std::shared_ptr<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
          problem(new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(
              adapter, opengv::sac_problems::absolute_pose::
                           AbsolutePoseSacProblem::KNEIP));
      ransac.sac_model_ = problem;
      ransac.threshold_ = config_.mapper_ransac_threshold;
      ransac.max_iterations_ = 100;
      ransac.computeModel();

      Eigen::Vector3d ransac_t =
          ransac.model_coefficients_.topRightCorner<3, 1>();
      std::cout << "              ransac_inliers=" << ransac.inliers_.size()
                << "  ransac_pose_t=[" << ransac_t.x() << ", " << ransac_t.y()
                << ", " << ransac_t.z() << "]" << std::endl;

      if ((int)ransac.inliers_.size() < config_.mapper_min_matches) {
        std::cout << "              RESULT: rejected (inliers "
                  << ransac.inliers_.size() << " < mapper_min_matches "
                  << config_.mapper_min_matches << ")" << std::endl;
        continue;
      }

      Sophus::SE3d T_partnerCam_newCam(
          ransac.model_coefficients_.topLeftCorner<3, 3>(),
          ransac.model_coefficients_.topRightCorner<3, 1>());

      have_loop = true;
      loop_partner_idx = partner_idx;
      T_body_partner_new =
          calib_.T_i_c[0] * T_partnerCam_newCam * calib_.T_i_c[0].inverse();

      std::cout << "              RESULT: ACCEPTED (loop closure #"
                << (num_loop_closures.load() + 1) << ")" << std::endl;
      break;  // first verified candidate wins -- keep it simple
    }
  }

  auto t3 = std::chrono::steady_clock::now();

  // --- Insert this keyframe + edges into the pose graph ---
  decomposeYPR(T_w_i_raw.rotationMatrix(), kf.roll, kf.pitch, kf.yaw);
  kf.t_opt = T_w_i_raw.translation();

  bool need_resolve = false;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);

    size_t new_idx = keyframes_.size();

    if (!keyframes_.empty()) {
      const LoopKeyframe& prev = keyframes_.back();
      Sophus::SE3d T_rel = prev.T_w_i_raw.inverse() * T_w_i_raw;

      PoseGraphEdge e;
      e.i = new_idx - 1;
      e.j = new_idx;
      e.dt = T_rel.translation();
      double r, p, y;
      decomposeYPR(T_rel.rotationMatrix(), r, p, y);
      e.dyaw = y;
      edges_.push_back(e);
    }

    if (have_loop) {
      PoseGraphEdge e;
      e.i = loop_partner_idx;
      e.j = new_idx;
      e.dt = T_body_partner_new.translation();
      double r, p, y;
      decomposeYPR(T_body_partner_new.rotationMatrix(), r, p, y);
      e.dyaw = y;
      edges_.push_back(e);

      num_loop_closures++;
      need_resolve = true;

      std::cout << "              (graph updated, total_closures="
                << num_loop_closures.load() << ")" << std::endl;
    }

    t_ns_to_idx_[kf_id] = new_idx;
    keyframes_.push_back(std::move(kf));
  }

  auto t4 = std::chrono::steady_clock::now();

  hash_bow_->add_to_database(TimeCamId(kf_id, 0),
                             keyframes_.back().kd0.bow_vector);

  auto t5 = std::chrono::steady_clock::now();

  if (need_resolve) solvePoseGraph();

  auto t6 = std::chrono::steady_clock::now();

  auto ms = [](std::chrono::steady_clock::time_point a,
              std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };

  std::cout << "[TIMING] kf=" << (keyframes_.size() - 1)
            << " detect_cam0=" << ms(t0, t1) << "ms"
            << " stereo_triangulate=" << ms(t1, t2) << "ms"
            << " loop_candidate_search=" << ms(t2, t3) << "ms"
            << " graph_insert=" << ms(t3, t4) << "ms"
            << " bow_add=" << ms(t4, t5) << "ms"
            << " pose_graph_solve=" << ms(t5, t6) << "ms"
            << " (n_keyframes=" << keyframes_.size()
            << " n_edges=" << edges_.size() << ")" << std::endl;
}

void OnlineLoopClosure::solvePoseGraph() {
  std::lock_guard<std::mutex> lock(state_mutex_);

  const size_t n = keyframes_.size();
  if (n < 2) return;

  // 4 unknowns per node (x,y,z,yaw), node 0 held fixed as the gauge anchor.
  const size_t num_free = n - 1;
  const size_t dim = 4 * num_free;

  auto param_offset = [&](size_t node_idx) -> int {
    return node_idx == 0 ? -1 : 4 * (int)(node_idx - 1);
  };

  double lambda = 1e-4;

  for (int iter = 0; iter < 15; iter++) {
    Eigen::MatrixXd JtJ = Eigen::MatrixXd::Zero((long)dim, (long)dim);
    Eigen::VectorXd Jtr = Eigen::VectorXd::Zero((long)dim);
    double total_err = 0;

    for (const auto& e : edges_) {
      const LoopKeyframe& ni = keyframes_[e.i];
      const LoopKeyframe& nj = keyframes_[e.j];

      Eigen::Matrix3d Ri = composeYPR(ni.roll, ni.pitch, ni.yaw);
      Eigen::Vector3d predicted_dt = Ri.transpose() * (nj.t_opt - ni.t_opt);
      double predicted_dyaw = wrapAngle(nj.yaw - ni.yaw);

      Eigen::Vector4d r;
      r.head<3>() = predicted_dt - e.dt;
      r(3) = wrapAngle(predicted_dyaw - e.dyaw);
      total_err += r.squaredNorm();

      // Numeric (central-difference) Jacobian, 4x4 per node, only for free
      // (non-anchor) nodes -- small graphs, correctness over cleverness.
      const double h = 1e-6;
      Eigen::Matrix<double, 4, 8> J = Eigen::Matrix<double, 4, 8>::Zero();

      auto eval = [&](double dxi, double dyi, double dzi, double dyawi,
                      double dxj, double dyj, double dzj, double dyawj) {
        Eigen::Vector3d ti = ni.t_opt + Eigen::Vector3d(dxi, dyi, dzi);
        Eigen::Vector3d tj = nj.t_opt + Eigen::Vector3d(dxj, dyj, dzj);
        Eigen::Matrix3d Ri2 = composeYPR(ni.roll, ni.pitch, ni.yaw + dyawi);
        Eigen::Vector4d rr;
        rr.head<3>() = Ri2.transpose() * (tj - ti) - e.dt;
        rr(3) = wrapAngle(wrapAngle(nj.yaw + dyawj - (ni.yaw + dyawi)) - e.dyaw);
        return rr;
      };

      for (int k = 0; k < 4; k++) {
        double d[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        d[k] = h;
        Eigen::Vector4d rp = eval(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
        d[k] = -h;
        Eigen::Vector4d rm = eval(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
        J.col(k) = (rp - rm) / (2 * h);
      }
      for (int k = 0; k < 4; k++) {
        double d[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        d[4 + k] = h;
        Eigen::Vector4d rp = eval(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
        d[4 + k] = -h;
        Eigen::Vector4d rm = eval(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
        J.col(4 + k) = (rp - rm) / (2 * h);
      }

      int oi = param_offset(e.i);
      int oj = param_offset(e.j);

      Eigen::Matrix<double, 4, 8> Jfull = J;

      auto accumulate = [&](int off_a, int block_a, int off_b, int block_b) {
        if (off_a < 0 || off_b < 0) return;
        JtJ.block(off_a, off_b, 4, 4) +=
            Jfull.block<4, 4>(0, block_a).transpose() *
            Jfull.block<4, 4>(0, block_b);
      };

      if (oi >= 0) {
        JtJ.block(oi, oi, 4, 4) += Jfull.block<4, 4>(0, 0).transpose() *
                                   Jfull.block<4, 4>(0, 0);
        Jtr.segment(oi, 4) += Jfull.block<4, 4>(0, 0).transpose() * r;
      }
      if (oj >= 0) {
        JtJ.block(oj, oj, 4, 4) += Jfull.block<4, 4>(0, 4).transpose() *
                                   Jfull.block<4, 4>(0, 4);
        Jtr.segment(oj, 4) += Jfull.block<4, 4>(0, 4).transpose() * r;
      }
      accumulate(oi, 0, oj, 4);
      if (oi >= 0 && oj >= 0) {
        JtJ.block(oj, oi, 4, 4) = JtJ.block(oi, oj, 4, 4).transpose();
      }
    }

    JtJ.diagonal().array() += lambda;

    Eigen::VectorXd dx = JtJ.ldlt().solve(-Jtr);
    if (!dx.allFinite()) break;

    for (size_t idx = 1; idx < n; idx++) {
      int off = param_offset(idx);
      keyframes_[idx].t_opt += dx.segment(off, 3);
      keyframes_[idx].yaw = wrapAngle(keyframes_[idx].yaw + dx(off + 3));
    }

    if (dx.norm() < 1e-7) break;
  }
}

Eigen::aligned_vector<Eigen::Vector3d> OnlineLoopClosure::getCorrectedTrajectory()
    const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  Eigen::aligned_vector<Eigen::Vector3d> out;
  out.reserve(keyframes_.size());
  for (const auto& kf : keyframes_) out.push_back(kf.t_opt);
  return out;
}

bool OnlineLoopClosure::getLatestCorrectedPose(Sophus::SE3d& out) const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (keyframes_.empty()) return false;
  const LoopKeyframe& kf = keyframes_.back();
  out = Sophus::SE3d(composeYPR(kf.roll, kf.pitch, kf.yaw), kf.t_opt);
  return true;
}

}  // namespace basalt
