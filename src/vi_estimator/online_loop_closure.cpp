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
#include <limits>

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <basalt/utils/keypoints.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#pragma GCC diagnostic pop

namespace basalt {

namespace {

// Looser than config_.mapper_max_hamming_distance / _second_best_test_ratio
// (70 / 1.2), used ONLY for matching between our own two calibrated
// cameras (cam0/cam1) at the same instant, never for matching across time.
// This is safe to loosen because, unlike temporal matching, a stereo match
// between a rigidly-mounted, calibrated pair can be verified exactly
// against the known relative geometry (see findInliersEssential() below) --
// so we can afford to let more candidate matches through here and let the
// geometry check reject the wrong ones, instead of relying on descriptor
// similarity alone to be strict enough. Goal: raise the fraction of each
// keyframe's corners that end up with a real triangulated 3D point (was
// consistently only 5-20% -- see conversation), so more temporal matches
// later have a chance of being PnP-ready.
constexpr int kStereoMaxHammingDistance = 90;
constexpr double kStereoSecondBestTestRatio = 1.5;
// Epipolar error threshold for findInliersEssential() -- same value
// Basalt's own offline mapper uses for the identical stereo-verification
// purpose (NfrMapper::match_stereo(), src/vi_estimator/nfr_mapper.cpp).
constexpr double kStereoEpipolarErrorThreshold = 1e-3;

// Keyframe quality gate: minimum triangulated 3D points a keyframe needs
// before it's allowed to become a future match TARGET (see processKeyframe()
// below). Basalt's own keyframe-selection trigger (sqrt_keypoint_vio.cpp,
// vio_new_kf_keypoints_thresh) is a pure motion/parallax heuristic -- it has
// no concept of image quality, so keyframes arrive with wildly inconsistent
// triangulation yield (observed anywhere from under 1% to 20%+ of detected
// corners across a real session).
//
// This must never go below mapper_min_matches (currently 13): a partner
// with fewer triangulated points than that can *never* produce enough
// PnP-ready matches to pass, no matter how good the descriptor matching is
// (pnp_ready_points is upper-bounded by the partner's own triangulated
// count), so anything below that floor is a pure waste, not a real chance.
//
// History: 30 (initial) -> 15 (too strict, excluded almost everything on
// Pi5) -> 25 (current). 15 let through partners whose triangulated points
// were sparse/marginal -- still numerically able to clear
// mapper_min_matches, but a thin or poorly-distributed point set can give
// RANSAC/PnP a badly-conditioned problem, producing a pose that passes
// verification while still being meaningfully wrong. A *wrong* accepted
// closure is worse than a missed one -- it actively pulls the trajectory
// off, rather than just failing to correct it. 25 is a deliberate partial
// revert to isolate whether this specific gate was the dominant cause of
// the increased drift observed after the previous set of loosenings,
// before touching the other thresholds. If Pi5 hardware instability keeps
// triangulation near-zero regardless, this alone won't fix it -- that's a
// data-quality problem no database threshold can compensate for.
constexpr int kMinTriangulatedPointsForDatabase = 25;

// Cap on how many BoW candidates get the FULL verification treatment
// (countMatchStages + matchDescriptors, both O(corners0 x corners_partner)
// brute-force Hamming searches, plus PnP-RANSAC) per keyframe. Measured
// live on a long (~90s, ~370-keyframe) session: loop_candidate_search cost
// grew from <1ms early on to 100-235ms once the database was rich enough
// that most keyframes had all mapper_num_frames_to_match (30) candidates
// clear the BoW threshold -- with ~450 corners/keyframe (since the
// epipolar-matching fix), verifying all 30 in the worst case (no early
// success) meant tens of millions of comparisons per keyframe, enough to
// exceed the ~230ms real keyframe arrival interval and make the background
// thread fall behind real time. BoW candidates are already score-sorted,
// so only trying the most-similar few first sacrifices little recall while
// bounding worst-case cost regardless of how large/rich the database gets.
constexpr size_t kMaxCandidatesToVerify = 5;

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

// Stereo-specific matcher that exploits the one piece of information plain
// matchDescriptors() doesn't use: for a rigidly-mounted, calibrated stereo
// pair, the relative pose is known exactly, so a true correspondence MUST
// lie on the known epipolar line. matchDescriptors() instead ranks
// candidates by descriptor similarity across the WHOLE other image and only
// checks epipolar geometry afterward (findInliersEssential) -- measured live
// on real data, this let ~420 corners per camera collapse to only ~43 raw
// matches and then just ~7 epipolar inliers: in a real room with
// repetitive-looking corners (tile grout, door frames, cable runs), the
// globally best-looking Hamming match is very often the WRONG corner
// elsewhere in the frame, which the TRUE corresponding corner (a worse but
// still legitimate Hamming score) then loses to and never gets a chance to
// be tried against. Restricting the candidate pool to epipolar-consistent
// corners BEFORE ranking by Hamming distance fixes that failure mode
// directly, instead of only discovering the loss after the fact.
void matchStereoEpipolar(const KeypointsData& kd0, const KeypointsData& kd1,
                         const Eigen::Matrix4d& E,
                         double epipolar_error_threshold, int hamming_threshold,
                         double ratio_threshold,
                         std::vector<std::pair<int, int>>& matches) {
  matches.clear();

  auto search = [&](const KeypointsData& a, const KeypointsData& b,
                    bool a_is_first, std::unordered_map<int, int>& out) {
    for (size_t i = 0; i < a.corner_descriptors.size(); i++) {
      int best_idx = -1, best_dist = 500, best2_dist = 500;
      for (size_t j = 0; j < b.corner_descriptors.size(); j++) {
        double epi_err =
            a_is_first
                ? std::abs(a.corners_3d[i].transpose() * E * b.corners_3d[j])
                : std::abs(b.corners_3d[j].transpose() * E * a.corners_3d[i]);
        if (epi_err > epipolar_error_threshold) continue;

        int dist =
            (int)(a.corner_descriptors[i] ^ b.corner_descriptors[j]).count();
        if (dist <= best_dist) {
          best2_dist = best_dist;
          best_dist = dist;
          best_idx = (int)j;
        } else if (dist < best2_dist) {
          best2_dist = dist;
        }
      }
      if (best_idx >= 0 && best_dist < hamming_threshold &&
          best_dist * ratio_threshold <= best2_dist) {
        out.emplace((int)i, best_idx);
      }
    }
  };

  std::unordered_map<int, int> m01, m10;
  search(kd0, kd1, true, m01);
  search(kd1, kd0, false, m10);

  for (const auto& kv : m01) {
    auto it = m10.find(kv.second);
    if (it != m10.end() && it->second == kv.first) {
      matches.emplace_back(kv.first, kv.second);
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
  localization_queue.set_capacity(1000);
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
    {
      std::vector<bool> success;
      calib_.intrinsics[1].unproject(kd1.corners, kd1.corners_3d, success);
    }

    Sophus::SE3d T_c0_c1 = calib_.T_i_c[0].inverse() * calib_.T_i_c[1];
    Eigen::Vector3d O1 = T_c0_c1.translation();

    // Epipolar-constrained matching (see matchStereoEpipolar() above) --
    // candidates are restricted to the known epipolar line BEFORE ranking
    // by descriptor similarity, rather than searching the whole image and
    // only checking geometry afterward. md.inliers == md.matches here by
    // construction, since every candidate already satisfied the epipolar
    // check during the search itself (unlike temporal/cross-time matching
    // below, which can't do this -- we don't know the relative pose
    // between two arbitrary past keyframes in advance, that's the whole
    // point of PnP-RANSAC there).
    Eigen::Matrix4d E;
    computeEssential(T_c0_c1, E);

    MatchData md;
    matchStereoEpipolar(kf.kd0, kd1, E, kStereoEpipolarErrorThreshold,
                        kStereoMaxHammingDistance, kStereoSecondBestTestRatio,
                        md.matches);
    md.inliers = md.matches;

    for (const auto& m : md.inliers) {
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

    // Diagnostic breakdown of *why* triangulated-point yield ends up where
    // it does -- distinguishes "not enough corners detected" (texture/
    // exposure problem) from "corners detected but stereo matching/
    // epipolar verification rejects them" (matching/calibration problem)
    // from "matches verified but triangulation itself fails" (cheirality/
    // range-gate problem), instead of only knowing the final count.
    std::cout << "[STEREO-DIAG] kf=" << keyframes_.size()
              << " corners0=" << kf.kd0.corners.size()
              << " corners1=" << kd1.corners.size()
              << " raw_matches=" << md.matches.size()
              << " epipolar_inliers=" << md.inliers.size()
              << " triangulated=" << kf.pts3d.size() << std::endl;
  }

  auto t2 = std::chrono::steady_clock::now();

  // --- Loop detection: query BoW DB restricted to earlier keyframes ---
  bool have_loop = false;
  size_t loop_partner_idx = 0;
  Sophus::SE3d T_body_partner_new;
  int loop_num_inliers = 0;

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

    // Only fully verify the top kMaxCandidatesToVerify by BoW score (see
    // constant comment above) -- results from querry_database() are only
    // guaranteed sorted when it truncated internally, so re-sort here to be
    // safe rather than assume that in all cases.
    std::sort(above_threshold.begin(), above_threshold.end(),
             [](const auto& a, const auto& b) { return a.second > b.second; });
    if (above_threshold.size() > kMaxCandidatesToVerify) {
      above_threshold.resize(kMaxCandidatesToVerify);
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

      // Reject on the RAW RANSAC inlier count *before* refining -- measured
      // live (~600-keyframe Pi5 session): opengv's optimize_nonlinear() is a
      // numeric-difference Levenberg-Marquardt optimizer (not analytic),
      // capped internally at 1000 function evaluations with very tight
      // tolerances, and its cost varies wildly with convergence difficulty
      // rather than point count (observed 73ms-455ms for similar inlier
      // counts, no clean correlation with N). Refining a candidate about to
      // be rejected anyway is pure waste, and the per-keyframe candidate
      // loop can attempt up to kMaxCandidatesToVerify of these -- so
      // checking the cheap raw count first, before paying for refinement,
      // bounds the common case to at most one refinement call per keyframe
      // (the loop breaks on first accepted candidate) instead of up to
      // kMaxCandidatesToVerify of them on failed attempts.
      if ((int)ransac.inliers_.size() < config_.mapper_min_matches) {
        std::cout << "              ransac_inliers=" << ransac.inliers_.size()
                  << " (raw, pre-refinement)" << std::endl;
        std::cout << "              RESULT: rejected (inliers "
                  << ransac.inliers_.size() << " < mapper_min_matches "
                  << config_.mapper_min_matches << ")" << std::endl;
        continue;
      }

      // Non-linear refinement over all RANSAC inliers -- mirrors the
      // pattern already used for the relative-pose case in
      // findInliersRansac() (src/utils/keypoints.cpp). RANSAC's model comes
      // from whichever minimal random subset scored best during search, not
      // a least-squares-optimal fit over every inlier; this refines it and
      // re-selects inliers against the refined model, same as the existing
      // relative-pose pattern.
      adapter.sett(ransac.model_coefficients_.topRightCorner<3, 1>());
      adapter.setR(ransac.model_coefficients_.topLeftCorner<3, 3>());
      opengv::transformation_t refined =
          opengv::absolute_pose::optimize_nonlinear(adapter, ransac.inliers_);
      ransac.sac_model_->selectWithinDistance(refined, ransac.threshold_,
                                              ransac.inliers_);
      ransac.model_coefficients_ = refined;

      Eigen::Vector3d ransac_t =
          ransac.model_coefficients_.topRightCorner<3, 1>();
      std::cout << "              ransac_inliers=" << ransac.inliers_.size()
                << " (refined)  ransac_pose_t=[" << ransac_t.x() << ", "
                << ransac_t.y() << ", " << ransac_t.z() << "]" << std::endl;

      if ((int)ransac.inliers_.size() < config_.mapper_min_matches) {
        std::cout << "              RESULT: rejected (refined inliers "
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

      // Publish the localization result immediately -- before the
      // pose-graph insertion/solve below -- so consumers that need "where
      // am I now" (navigation, RTL) don't have to wait on the potentially
      // slower global solve. Composed against the reference keyframe's
      // current CORRECTED pose (already reflecting any earlier loop
      // closures), not its raw VIO pose, so T_w_current is already
      // globally-consistent without needing a fresh solve first.
      {
        LocalizationResult loc;
        loc.t_ns = kf_id;
        loc.reference_t_ns = partner_t_ns;
        loc.T_reference_current = T_body_partner_new;
        Sophus::SE3d T_w_reference_corrected(
            composeYPR(partner->roll, partner->pitch, partner->yaw),
            partner->t_opt);
        loc.T_w_current = T_w_reference_corrected * T_body_partner_new;
        loc.num_inliers = (int)ransac.inliers_.size();
        loop_num_inliers = loc.num_inliers;
        localization_queue.try_push(loc);
      }

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
      // See PoseGraphEdge::weight comment (online_loop_closure.h) -- a
      // closure that just barely cleared mapper_min_matches gets the same
      // baseline trust as an odometry edge (weight 1.0); one with several
      // times as many inliers pulls proportionally harder, capped at 5x so
      // a single very-strong match can't dominate the graph unboundedly.
      e.weight = std::clamp(
          loop_num_inliers / std::max(1.0, config_.mapper_min_matches), 1.0,
          5.0);
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

  // Quality gate (see kMinTriangulatedPointsForDatabase above): this
  // keyframe still got a pose-graph node and a chance to QUERY against
  // history above (we don't get to choose which frames Basalt hands us),
  // but only keyframes with enough triangulated points are added as future
  // match TARGETS, so a poor one can't poison later keyframes' candidate
  // pool the way we saw happen repeatedly during live testing.
  //
  // Distance gate (config_.mapper_min_keyframe_storage_dist, Step 4 of the
  // visual-localization/RTL proposal): a keyframe that clears the quality
  // gate must ALSO be far enough (raw VIO estimate) from the last STORED
  // keyframe to be added. Without this, keyframes get stored roughly every
  // ~230ms of motion (near-continuous), producing a database far denser
  // than needed for matching and, more importantly for RTL, too large to
  // route waypoints through. Caveat: this measures distance from the raw,
  // potentially-drifting VIO estimate -- acceptable here because it's a
  // LOCAL delta since the last stored keyframe (seconds/meters), not a
  // global position, so accumulated drift over that short an interval
  // should be small; worth confirming empirically rather than assuming.
  size_t num_pts3d = keyframes_.back().pts3d.size();
  bool quality_ok = (int)num_pts3d >= kMinTriangulatedPointsForDatabase;
  double dist_since_last_stored =
      has_stored_any_
          ? (T_w_i_raw.translation() - last_stored_raw_position_).norm()
          : std::numeric_limits<double>::infinity();
  bool distance_ok =
      !has_stored_any_ ||
      dist_since_last_stored >= config_.mapper_min_keyframe_storage_dist;

  if (quality_ok && distance_ok) {
    hash_bow_->add_to_database(TimeCamId(kf_id, 0),
                               keyframes_.back().kd0.bow_vector);
    last_stored_raw_position_ = T_w_i_raw.translation();
    has_stored_any_ = true;
  } else {
    std::cout << "[ONLINE-LOOP] kf=" << (keyframes_.size() - 1)
              << " excluded from candidate database (";
    if (!quality_ok) {
      std::cout << num_pts3d << " triangulated points < "
                << kMinTriangulatedPointsForDatabase;
    }
    if (!quality_ok && !distance_ok) std::cout << "; ";
    if (!distance_ok) {
      std::cout << dist_since_last_stored << "m since last stored < "
                << config_.mapper_min_keyframe_storage_dist << "m";
    }
    std::cout << ")" << std::endl;
  }

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

  // Generator matrix for d Rz(yaw)/d yaw = Rz(yaw) * skewZ (standard result
  // for a single-axis rotation derivative).
  Eigen::Matrix3d skewZ;
  skewZ << 0, -1, 0, 1, 0, 0, 0, 0, 0;

  double lambda = 1e-4;

  for (int iter = 0; iter < 15; iter++) {
    // Sparse + analytic-Jacobian rebuild of the normal equations. Profiling
    // (see conversation) showed the previous dense (Eigen::MatrixXd) +
    // numeric-Jacobian (central-difference) version taking 1+ second per
    // solve at only ~350 nodes -- almost entirely from rebuilding and
    // factorizing a dense dim x dim matrix from scratch every iteration.
    // The graph is naturally sparse (each node only touches a couple of
    // edges), so a triplet-built SparseMatrix + SimplicialLDLT scales with
    // the number of actual connections instead of dim^2/dim^3. Analytic
    // Jacobians (derived from d(R^T)/d(yaw) = [d Rz(yaw)/d yaw * Ry * Rx]^T)
    // remove the 16 extra residual evaluations per edge the numeric version
    // needed, though that was a much smaller share of the 1s+ cost.
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(edges_.size() * 40);
    Eigen::VectorXd Jtr = Eigen::VectorXd::Zero((long)dim);

    auto add_block = [&](int row_off, int col_off,
                         const Eigen::Matrix4d& block) {
      for (int r_ = 0; r_ < 4; r_++)
        for (int c_ = 0; c_ < 4; c_++)
          if (block(r_, c_) != 0.0)
            triplets.emplace_back(row_off + r_, col_off + c_, block(r_, c_));
    };

    for (const auto& e : edges_) {
      const LoopKeyframe& ni = keyframes_[e.i];
      const LoopKeyframe& nj = keyframes_[e.j];

      // Ri = Rz(yaw_i) * Ci, where Ci = Ry(pitch_i) * Rx(roll_i) is
      // constant (roll/pitch are never optimized) -- matches composeYPR().
      Eigen::Matrix3d Ci = composeYPR(ni.roll, ni.pitch, 0.0);
      Eigen::Matrix3d Rzi =
          Eigen::AngleAxisd(ni.yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
      Eigen::Matrix3d Ri = Rzi * Ci;

      Eigen::Vector3d dt_vec = nj.t_opt - ni.t_opt;
      Eigen::Vector3d predicted_dt = Ri.transpose() * dt_vec;
      double predicted_dyaw = wrapAngle(nj.yaw - ni.yaw);

      Eigen::Vector4d r;
      r.head<3>() = predicted_dt - e.dt;
      r(3) = wrapAngle(predicted_dyaw - e.dyaw);

      // d(Ri^T)/d(yaw_i) = [Rzi * skewZ * Ci]^T, so
      // d(predicted_dt)/d(yaw_i) = [Rzi * skewZ * Ci]^T * (tj - ti).
      Eigen::Matrix3d dRi_dyaw = Rzi * skewZ * Ci;
      Eigen::Vector3d dt_dyawi = dRi_dyaw.transpose() * dt_vec;

      Eigen::Matrix<double, 4, 8> J = Eigen::Matrix<double, 4, 8>::Zero();
      J.block<3, 3>(0, 0) = -Ri.transpose();  // d r_t / d t_i
      J.block<3, 1>(0, 3) = dt_dyawi;         // d r_t / d yaw_i
      J.block<3, 3>(0, 4) = Ri.transpose();   // d r_t / d t_j
      // d r_t / d yaw_j == 0 (t_j doesn't rotate through node j's yaw)
      J(3, 3) = -1.0;  // d r_yaw / d yaw_i
      J(3, 7) = 1.0;   // d r_yaw / d yaw_j

      int oi = param_offset(e.i);
      int oj = param_offset(e.j);

      // Scaling the per-edge contribution by e.weight before accumulation
      // is equivalent to minimizing sum_e weight_e * ||r_e||^2 -- the
      // normal equations become sum_e w_e * J_e^T J_e * dx = -sum_e w_e *
      // J_e^T r_e, same Gauss-Newton derivation as before, just weighted.
      Eigen::Matrix4d Jii =
          e.weight * J.block<4, 4>(0, 0).transpose() * J.block<4, 4>(0, 0);
      Eigen::Matrix4d Jjj =
          e.weight * J.block<4, 4>(0, 4).transpose() * J.block<4, 4>(0, 4);
      Eigen::Matrix4d Jij =
          e.weight * J.block<4, 4>(0, 0).transpose() * J.block<4, 4>(0, 4);

      if (oi >= 0) {
        add_block(oi, oi, Jii);
        Jtr.segment(oi, 4) += e.weight * J.block<4, 4>(0, 0).transpose() * r;
      }
      if (oj >= 0) {
        add_block(oj, oj, Jjj);
        Jtr.segment(oj, 4) += e.weight * J.block<4, 4>(0, 4).transpose() * r;
      }
      if (oi >= 0 && oj >= 0) {
        add_block(oi, oj, Jij);
        add_block(oj, oi, Jij.transpose());
      }
    }

    for (size_t k = 0; k < dim; k++)
      triplets.emplace_back((int)k, (int)k, lambda);

    Eigen::SparseMatrix<double> JtJ((long)dim, (long)dim);
    JtJ.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(JtJ);
    if (solver.info() != Eigen::Success) break;

    Eigen::VectorXd dx = solver.solve(-Jtr);
    if (solver.info() != Eigen::Success || !dx.allFinite()) break;

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

void OnlineLoopClosure::getCorrectedTrajectoryWithTimestamps(
    std::vector<int64_t>& t_ns,
    Eigen::aligned_vector<Eigen::Vector3d>& positions) const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  t_ns.clear();
  positions.clear();
  t_ns.reserve(keyframes_.size());
  positions.reserve(keyframes_.size());
  for (const auto& kf : keyframes_) {
    t_ns.push_back(kf.t_ns);
    positions.push_back(kf.t_opt);
  }
}

bool OnlineLoopClosure::getLatestCorrectedPose(Sophus::SE3d& out) const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (keyframes_.empty()) return false;
  const LoopKeyframe& kf = keyframes_.back();
  out = Sophus::SE3d(composeYPR(kf.roll, kf.pitch, kf.yaw), kf.t_opt);
  return true;
}

}  // namespace basalt
