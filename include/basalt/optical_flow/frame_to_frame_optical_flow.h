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

#pragma once

#include <deque>
#include <mutex>
#include <thread>

#include <sophus/se2.hpp>
#include <sophus/se3.hpp>

#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/parallel_for.h>

#include <basalt/optical_flow/optical_flow.h>
#include <basalt/optical_flow/patch.h>

#include <basalt/image/image_pyr.h>
#include <basalt/utils/keypoints.h>

namespace basalt {

/// Unlike PatchOpticalFlow, FrameToFrameOpticalFlow always tracks patches
/// against the previous frame, not the initial frame where a track was created.
/// While it might cause more drift of the patch location, it leads to longer
/// tracks in practice.
template <typename Scalar, template <typename> typename Pattern>
class FrameToFrameOpticalFlow : public OpticalFlowBase {
 public:
  typedef OpticalFlowPatch<Scalar, Pattern<Scalar>> PatchT;

  typedef Eigen::Matrix<Scalar, 2, 1> Vector2;
  typedef Eigen::Matrix<Scalar, 2, 2> Matrix2;

  typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
  typedef Eigen::Matrix<Scalar, 3, 3> Matrix3;

  typedef Eigen::Matrix<Scalar, 4, 1> Vector4;
  typedef Eigen::Matrix<Scalar, 4, 4> Matrix4;

  typedef Sophus::SE2<Scalar> SE2;

  FrameToFrameOpticalFlow(const VioConfig& config,
                          const basalt::Calibration<double>& calib)
      : t_ns(-1), frame_counter(0), last_keypoint_id(0), config(config) {
    input_queue.set_capacity(10);

    this->calib = calib.cast<Scalar>();

    patch_coord = PatchT::pattern2.template cast<float>();

    if (calib.intrinsics.size() > 1) {
      Eigen::Matrix4d Ed;
      Sophus::SE3d T_i_j = calib.T_i_c[0].inverse() * calib.T_i_c[1];
      computeEssential(T_i_j, Ed);
      E = Ed.cast<Scalar>();

      // See trackNewPointsStereoWithDepthSeeds() -- maps a point in cam0's
      // frame to cam1's frame for depth-hypothesis reprojection seeding.
      T_c1_c0_ = (this->calib.T_i_c[1].inverse() * this->calib.T_i_c[0]);
    }

    processing_thread.reset(
        new std::thread(&FrameToFrameOpticalFlow::processingLoop, this));
  }

  ~FrameToFrameOpticalFlow() { processing_thread->join(); }

  // Feeds the rotation-compensated KLT seed (see trackPoints()'s comment
  // for the full reasoning) -- called from whatever thread owns the raw
  // IMU tap (see OakDDevice::setImuTapQueue()'s comment), NOT from
  // processingLoop()'s thread, so this only ever touches gyro_queue_
  // under gyro_mutex_. Stores raw IMU-frame samples rather than
  // converting to any one camera's frame here, since a stereo rig tracks
  // two cameras with different (if similar) extrinsics -- the per-camera
  // conversion happens lazily in integrateRotation() instead.
  void addIMUToQueue(const ImuData<double>::Ptr& data) override {
    if (!data) return;

    std::lock_guard<std::mutex> lock(gyro_mutex_);
    gyro_queue_.emplace_back(data->t_ns, data->gyro);

    // Trim to a bounded window -- comfortably more than the 1-2 frame
    // intervals integrateRotation() ever actually needs (30fps => ~33ms
    // apart), just generous enough to tolerate some jitter in when frames
    // vs. IMU samples arrive relative to each other.
    while (!gyro_queue_.empty() &&
           data->t_ns - gyro_queue_.front().first > kGyroHistoryNs) {
      gyro_queue_.pop_front();
    }
  }

  void processingLoop() {
    OpticalFlowInput::Ptr input_ptr;

    while (true) {
      input_queue.pop(input_ptr);

      if (!input_ptr.get()) {
        if (output_queue) output_queue->push(nullptr);
        break;
      }

      processFrame(input_ptr->t_ns, input_ptr);
    }
  }

  void processFrame(int64_t curr_t_ns, OpticalFlowInput::Ptr& new_img_vec) {
    for (const auto& v : new_img_vec->img_data) {
      if (!v.img.get()) return;
    }

    if (t_ns < 0) {
      t_ns = curr_t_ns;

      transforms.reset(new OpticalFlowResult);
      transforms->observations.resize(calib.intrinsics.size());
      transforms->t_ns = t_ns;

      pyramid.reset(new std::vector<basalt::ManagedImagePyr<uint16_t>>);
      pyramid->resize(calib.intrinsics.size());

      tbb::parallel_for(tbb::blocked_range<size_t>(0, calib.intrinsics.size()),
                        [&](const tbb::blocked_range<size_t>& r) {
                          for (size_t i = r.begin(); i != r.end(); ++i) {
                            pyramid->at(i).setFromImage(
                                *new_img_vec->img_data[i].img,
                                config.optical_flow_levels);
                          }
                        });

      transforms->input_images = new_img_vec;

      addPoints();
      filterPoints();

    } else {
      t_ns = curr_t_ns;

      old_pyramid = pyramid;

      pyramid.reset(new std::vector<basalt::ManagedImagePyr<uint16_t>>);
      pyramid->resize(calib.intrinsics.size());
      tbb::parallel_for(tbb::blocked_range<size_t>(0, calib.intrinsics.size()),
                        [&](const tbb::blocked_range<size_t>& r) {
                          for (size_t i = r.begin(); i != r.end(); ++i) {
                            pyramid->at(i).setFromImage(
                                *new_img_vec->img_data[i].img,
                                config.optical_flow_levels);
                          }
                        });

      OpticalFlowResult::Ptr new_transforms;
      new_transforms.reset(new OpticalFlowResult);
      new_transforms->observations.resize(calib.intrinsics.size());
      new_transforms->t_ns = t_ns;

      static const Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f>
          kEmptyTransformMap;

      int64_t t_ns_0 = prev_transforms ? prev_transforms->t_ns : -1;

      for (size_t i = 0; i < calib.intrinsics.size(); i++) {
        trackPoints(old_pyramid->at(i), pyramid->at(i),
                    prev_transforms ? prev_transforms->observations[i]
                                     : kEmptyTransformMap,
                    transforms->observations[i],
                    new_transforms->observations[i], i, t_ns_0,
                    transforms->t_ns, t_ns);
      }

      // 'transforms' (this call's N-1) becomes N-2 for the NEXT call,
      // before it's overwritten below -- see prev_transforms's comment.
      prev_transforms = transforms;
      transforms = new_transforms;
      transforms->input_images = new_img_vec;

      addPoints();
      filterPoints();
    }

    if (output_queue && frame_counter % config.optical_flow_skip_frames == 0) {
      output_queue->push(transforms);
    }

    frame_counter++;
  }

  // Sums buffered gyro samples in (t0_ns, t1_ns] into a first-order-
  // integrated incremental rotation, converted from IMU frame into
  // cam_id's own frame first (a rigid, constant rotation between the two,
  // so angular velocity transforms by that rotation alone -- no lever-arm
  // term needed, unlike linear velocity). Returns identity if no samples
  // fall in that window (IMU tap not wired up yet, a startup gap, or
  // t0_ns/t1_ns out of range) -- same "harmless no-op fallback" as every
  // other optional signal in this codebase, since the caller's seed then
  // just falls back to the un-rotated pixel.
  Sophus::SO3<Scalar> integrateRotation(int64_t t0_ns, int64_t t1_ns,
                                        size_t cam_id) const {
    if (t0_ns < 0 || t1_ns <= t0_ns || cam_id >= calib.T_i_c.size())
      return Sophus::SO3<Scalar>();

    // R_ic: rotation from camera cam_id's frame into the IMU frame.
    // omega_imu transforms into cam_id's frame via its inverse.
    Sophus::SO3<Scalar> R_ic = calib.T_i_c[cam_id].so3();

    std::lock_guard<std::mutex> lock(gyro_mutex_);

    Vector3 accumulated = Vector3::Zero();
    int64_t prev_t = t0_ns;
    for (const auto& kv : gyro_queue_) {
      if (kv.first <= t0_ns) continue;
      if (kv.first > t1_ns) break;

      Scalar dt = Scalar((kv.first - prev_t) / 1e9);
      Vector3 omega_imu = kv.second.template cast<Scalar>();
      Vector3 omega_cam = R_ic.inverse() * omega_imu;
      accumulated += omega_cam * dt;
      prev_t = kv.first;
    }

    return Sophus::SO3<Scalar>::exp(accumulated);
  }

  // Reprojects `pixel` through a pure rotation applied to its unprojected
  // ray, predicting where a STATIC world point last seen at `pixel` will
  // appear after the camera rotates by R_delta (R_delta in the same
  // body-frame convention as integrateRotation()/preintegration.h:
  // R_world_cam(N) = R_world_cam(N-1) * R_delta). For a fixed point,
  // R_world_cam(N-1)*p_cam(N-1) = R_world_cam(N)*p_cam(N), which solves to
  // p_cam(N) = R_delta.inverse() * p_cam(N-1) -- so this applies the
  // INVERSE of R_delta, not R_delta itself; getting this backwards would
  // predict motion in the wrong direction rather than just not helping.
  // Depth-independent (scaling a ray's length along the same direction
  // doesn't change where it projects), so this needs no depth hypothesis,
  // unlike trackNewPointsStereoWithDepthSeeds(). Falls back to returning
  // `pixel` unchanged if unproject/project fails (e.g. pixel outside the
  // calibrated FOV after rotation) -- same fallback style as this file's
  // other seeding helpers.
  Vector2 rotateCompensatedPixel(const Vector2& pixel,
                                 const Sophus::SO3<Scalar>& R_delta,
                                 size_t cam_id) const {
    Vector4 ray;
    if (!calib.intrinsics[cam_id].unproject(pixel, ray)) return pixel;

    Sophus::SO3<Scalar> R_delta_inv = R_delta.inverse();
    Vector4 rotated;
    rotated.template head<3>() = R_delta_inv * ray.template head<3>();
    rotated[3] = ray[3];

    Vector2 out;
    if (!calib.intrinsics[cam_id].project(rotated, out)) return pixel;
    return out;
  }

  // transform_map_0 is two frames back (N-2). t_ns_0/t_ns_1/t_ns_2 are
  // the N-2/N-1/N frame timestamps -- used together with transform_map_0
  // to predict each track's seed below. See seed_vec's comment.
  void trackPoints(const basalt::ManagedImagePyr<uint16_t>& pyr_1,
                   const basalt::ManagedImagePyr<uint16_t>& pyr_2,
                   const Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f>&
                       transform_map_0,
                   const Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f>&
                       transform_map_1,
                   Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f>&
                       transform_map_2,
                   size_t cam_id, int64_t t_ns_0, int64_t t_ns_1,
                   int64_t t_ns_2) const {
    size_t num_points = transform_map_1.size();

    std::vector<KeypointId> ids;
    Eigen::aligned_vector<Eigen::AffineCompact2f> init_vec;
    // Predicted initial guess for trackPoint()'s search -- previously
    // this was always transform_1 unchanged (zero-motion seed). At high
    // speed/low FPS, frame-to-frame pixel displacement can exceed KLT's
    // small per-level convergence basin before the search even starts,
    // which is the dominant cause of tracked-point collapse during fast
    // motion (see this file's stereo depth-hypothesis seeding below for
    // the same "seed, not tracker, is the bottleneck" issue already
    // solved there).
    //
    // Two components, combined so neither double-counts the other:
    // 1. Rotation (R_1_to_2, from buffered gyro -- see integrateRotation())
    //    applied to transform_1's pixel. Depth-independent and reacts
    //    instantly to the CURRENT measured angular rate, unlike a vision-
    //    history-based estimate, which would lag behind an accelerating
    //    pan. Live-motivated (2026-09-28, pi5-092377b8): freezes during a
    //    "walk around and look at things" test correlated with gyro rates
    //    up to 2-4.6 rad/s, which pure translational seeding (the first
    //    version of this mechanism) doesn't compensate for at all.
    // 2. Residual translation, estimated from the track's own last-two-
    //    frame displacement AFTER de-rotating frame N-2 into frame N-1's
    //    frame (via R_0_to_1) -- isolates the parallax/translation-driven
    //    component so it isn't counted twice (once implicitly via the
    //    raw vision history, once explicitly via the rotation term).
    //
    // Falls back gracefully component-by-component: no gyro data yet ->
    // integrateRotation() returns identity (rotation term is a no-op);
    // no transform_map_0 entry (freshly added point, or first tracked
    // frame) -> residual translation is zero. With both absent this is
    // exactly the old zero-motion seed. Unproven against a live retest --
    // second version of this mechanism, first with rotation compensation.
    Eigen::aligned_vector<Eigen::AffineCompact2f> seed_vec;

    ids.reserve(num_points);
    init_vec.reserve(num_points);
    seed_vec.reserve(num_points);

    Sophus::SO3<Scalar> R_1_to_2 = integrateRotation(t_ns_1, t_ns_2, cam_id);
    bool have_prev_frame = t_ns_0 >= 0;
    Sophus::SO3<Scalar> R_0_to_1 =
        have_prev_frame ? integrateRotation(t_ns_0, t_ns_1, cam_id)
                        : Sophus::SO3<Scalar>();

    for (const auto& kv : transform_map_1) {
      ids.push_back(kv.first);
      init_vec.push_back(kv.second);

      Eigen::AffineCompact2f seed = kv.second;
      seed.translation() =
          rotateCompensatedPixel(kv.second.translation(), R_1_to_2, cam_id);

      auto it0 = transform_map_0.find(kv.first);
      if (have_prev_frame && it0 != transform_map_0.end()) {
        Vector2 rotated_n2 = rotateCompensatedPixel(
            it0->second.translation(), R_0_to_1, cam_id);
        seed.translation() += (kv.second.translation() - rotated_n2);
      }
      seed_vec.push_back(seed);
    }

    tbb::concurrent_unordered_map<
        KeypointId, Eigen::AffineCompact2f, std::hash<KeypointId>,
        std::equal_to<KeypointId>,
        Eigen::aligned_allocator<
            std::pair<const KeypointId, Eigen::AffineCompact2f>>>
        result;

    auto compute_func = [&](const tbb::blocked_range<size_t>& range) {
      for (size_t r = range.begin(); r != range.end(); ++r) {
        const KeypointId id = ids[r];

        const Eigen::AffineCompact2f& transform_1 = init_vec[r];
        Eigen::AffineCompact2f transform_2 = seed_vec[r];

        bool valid = trackPoint(pyr_1, pyr_2, transform_1, transform_2);

        if (valid) {
          // Seed the backward track at the old position shifted by the
          // forward search's own correction to its seed. Starting it at
          // transform_2 (new-frame coordinates) put it a whole frame's
          // motion away from the answer -- ~40 px at 200 deg/s -- so fast
          // turns failed the round trip and correct tracks were dropped.
          Eigen::AffineCompact2f transform_1_recovered = transform_2;
          transform_1_recovered.translation() =
              transform_1.translation() +
              (transform_2.translation() - seed_vec[r].translation());

          valid = trackPoint(pyr_2, pyr_1, transform_2, transform_1_recovered);

          if (valid) {
            Scalar dist2 = (transform_1.translation() -
                            transform_1_recovered.translation())
                               .squaredNorm();

            if (dist2 < config.optical_flow_max_recovered_dist2) {
              result[id] = transform_2;
            }
          }
        }
      }
    };

    tbb::blocked_range<size_t> range(0, num_points);

    tbb::parallel_for(range, compute_func);
    // compute_func(range);

    transform_map_2.clear();
    transform_map_2.insert(result.begin(), result.end());
  }

  // Stereo-specific seeding for newly-detected cam0 corners -- unlike
  // trackPoints() above (correct for frame-to-frame TEMPORAL tracking,
  // where near-zero motion between consecutive frames makes "start the
  // search at the same pixel" a good initial guess), a same-instant
  // cam0->cam1 pair can have substantial pixel disparity even for a
  // perfectly static scene, and KLT's small per-level search window can't
  // recover a correspondence that starts that far from the true target.
  // Measured live (see conversation): the naive same-pixel seed (i.e.
  // calling trackPoints() here the same way as for temporal tracking)
  // succeeded on only ~1.4% of newly-detected corners, vs. 74.7% of the
  // ones that DID get a seed close enough to converge -- the seed, not
  // the tracker itself, was the bottleneck. Tries a small set of
  // representative scene depths (covering this rig's ~0.3-6m intended
  // operating range), reprojects each cam0 ray into cam1 using the
  // calibrated extrinsics to get a much better starting pixel, and keeps
  // the first depth hypothesis that produces a forward-backward-
  // consistent track -- same acceptance criteria as trackPoints() above,
  // just a smarter initial guess.
  void trackNewPointsStereoWithDepthSeeds(
      const basalt::ManagedImagePyr<uint16_t>& pyr0,
      const basalt::ManagedImagePyr<uint16_t>& pyr1,
      const Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f>&
          new_poses0,
      Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f>& new_poses1)
      const {
    // Configurable via config.optical_flow_stereo_seed_depths_m (see that
    // field's comment) -- was a fixed 7-depth set {0.3, 0.6, 1.0, 1.5,
    // 2.5, 4.0, 6.0}, widened from an original {0.5, 1.0, 2.0, 4.0} as
    // one of two levers behind the post-rotation "tracked points take
    // 1-2s to come back" recovery lag (see conversation): landmark-pool
    // rebuild speed after motion blur wipes it out is capped partly by
    // how many of a keyframe's new corners this seeding successfully
    // triangulates. More/finer depths raise that per-keyframe yield
    // directly (run 20260921_134359: 1.37% -> 2.80% klt_stereo_ok going
    // from 0 to 4 depths), but each one is a real added cost per new
    // corner -- self-limiting in that it only applies to newly-detected
    // corners, but NOT free, and a compute-constrained platform running
    // the same corner-density settings as the laptop this was tuned on
    // may need fewer of them (see the config field's comment for why
    // this became configurable rather than a fixed constant).
    const std::vector<double>& seed_depths_m =
        config.optical_flow_stereo_seed_depths_m;

    std::vector<KeypointId> ids;
    Eigen::aligned_vector<Eigen::AffineCompact2f> init_vec;
    ids.reserve(new_poses0.size());
    init_vec.reserve(new_poses0.size());
    for (const auto& kv : new_poses0) {
      ids.push_back(kv.first);
      init_vec.push_back(kv.second);
    }

    tbb::concurrent_unordered_map<
        KeypointId, Eigen::AffineCompact2f, std::hash<KeypointId>,
        std::equal_to<KeypointId>,
        Eigen::aligned_allocator<
            std::pair<const KeypointId, Eigen::AffineCompact2f>>>
        result;

    // Per-corner outcome of the best stage reached across all seed depths:
    // 0 no seed projected, 1 forward KLT failed, 2 backward KLT failed,
    // 3 round trip too large, 4 matched. Plus each round-trip-rejected
    // corner's smallest round-trip error (px), to size the tolerance.
    std::vector<int> outcome(ids.size(), 0);
    std::vector<float> best_roundtrip_px(ids.size(), -1.f);

    auto compute_func = [&](const tbb::blocked_range<size_t>& range) {
      for (size_t r = range.begin(); r != range.end(); ++r) {
        const KeypointId id = ids[r];
        const Eigen::AffineCompact2f& transform_1 = init_vec[r];

        Vector4 ray0;
        if (!calib.intrinsics[0].unproject(transform_1.translation(), ray0))
          continue;
        Vector3 dir0 = ray0.template head<3>().normalized();

        // KLT forward + backward from one seed; true if the round trip closes.
        auto try_seed = [&](const Vector2& seed_px) {
          Eigen::AffineCompact2f transform_2;
          transform_2.setIdentity();
          transform_2.translation() = seed_px;
          outcome[r] = std::max(outcome[r], 1);
          if (!trackPoint(pyr0, pyr1, transform_1, transform_2)) return false;
          outcome[r] = std::max(outcome[r], 2);
          // Backward check: start at the corner itself and refine at full
          // resolution only. Starting from cam1 coordinates (a full
          // disparity away) or from the seed-corrected position (which
          // mirrors the depth guess's error), or passing through the coarse
          // levels, pulled correct matches off; on real frames this check
          // alone rejected most of them.
          PatchT patch1(pyr1.lvl(0), transform_2.translation());
          if (!patch1.valid) return false;
          Eigen::AffineCompact2f transform_1_recovered = transform_1;
          transform_1_recovered.linear().setIdentity();
          if (!trackPointAtLevel(pyr0.lvl(0), patch1, transform_1_recovered))
            return false;
          outcome[r] = std::max(outcome[r], 3);
          Scalar dist2 = (transform_1.translation() -
                          transform_1_recovered.translation())
                             .squaredNorm();
          float dist_px = std::sqrt(float(dist2));
          if (best_roundtrip_px[r] < 0.f || dist_px < best_roundtrip_px[r])
            best_roundtrip_px[r] = dist_px;
          // Matches between two cameras round-trip at ~0.2-0.3 px even when
          // correct (the lenses distort slightly differently), so the 0.2 px
          // same-camera tolerance rejected about a third of good ones; the
          // epipolar check in filterPoints() still removes wrong matches.
          constexpr Scalar kStereoMaxRecoveredDist2 = 0.25;  // 0.5 px
          if (dist2 < kStereoMaxRecoveredDist2) {
            result[id] = transform_2;
            outcome[r] = 4;
            return true;
          }
          return false;
        };

        for (double depth_d : seed_depths_m) {
          Vector3 pt_c1 = T_c1_c0_ * (dir0 * Scalar(depth_d));
          Vector4 pt_c1_h;
          pt_c1_h.template head<3>() = pt_c1;
          pt_c1_h[3] = Scalar(1);
          Vector2 seed_px;
          if (!calib.intrinsics[1].project(pt_c1_h, seed_px)) continue;
          if (try_seed(seed_px)) break;  // this depth hypothesis worked
        }
      }
    };

    tbb::blocked_range<size_t> range(0, ids.size());
    tbb::parallel_for(range, compute_func);

    new_poses1.clear();
    new_poses1.insert(result.begin(), result.end());

    if (!ids.empty()) {
      int n[5] = {0, 0, 0, 0, 0};
      std::vector<float> rejected_rt;
      for (size_t r = 0; r < ids.size(); r++) {
        n[outcome[r]]++;
        if (outcome[r] == 3) rejected_rt.push_back(best_roundtrip_px[r]);
      }
      float med = -1.f;
      if (!rejected_rt.empty()) {
        std::nth_element(rejected_rt.begin(), rejected_rt.begin() + rejected_rt.size() / 2,
                         rejected_rt.end());
        med = rejected_rt[rejected_rt.size() / 2];
      }
      std::cout << "[STEREO-MATCH-DIAG] corners=" << ids.size()
                << " no_seed=" << n[0] << " fwd_fail=" << n[1]
                << " bwd_fail=" << n[2] << " roundtrip_reject=" << n[3]
                << " matched=" << n[4]
                << " roundtrip_reject_median_px=" << med << std::endl;
    }
  }

  inline bool trackPoint(const basalt::ManagedImagePyr<uint16_t>& old_pyr,
                         const basalt::ManagedImagePyr<uint16_t>& pyr,
                         const Eigen::AffineCompact2f& old_transform,
                         Eigen::AffineCompact2f& transform) const {
    bool patch_valid = true;

    transform.linear().setIdentity();

    for (int level = config.optical_flow_levels; level >= 0 && patch_valid;
         level--) {
      const Scalar scale = 1 << level;

      transform.translation() /= scale;

      PatchT p(old_pyr.lvl(level), old_transform.translation() / scale);

      patch_valid &= p.valid;
      if (patch_valid) {
        // Perform tracking on current level
        patch_valid &= trackPointAtLevel(pyr.lvl(level), p, transform);
      }

      transform.translation() *= scale;
    }

    transform.linear() = old_transform.linear() * transform.linear();

    return patch_valid;
  }

  inline bool trackPointAtLevel(const Image<const uint16_t>& img_2,
                                const PatchT& dp,
                                Eigen::AffineCompact2f& transform) const {
    bool patch_valid = true;

    for (int iteration = 0;
         patch_valid && iteration < config.optical_flow_max_iterations;
         iteration++) {
      typename PatchT::VectorP res;

      typename PatchT::Matrix2P transformed_pat =
          transform.linear().matrix() * PatchT::pattern2;
      transformed_pat.colwise() += transform.translation();

      patch_valid &= dp.residual(img_2, transformed_pat, res);

      if (patch_valid) {
        const Vector3 inc = -dp.H_se2_inv_J_se2_T * res;

        // avoid NaN in increment (leads to SE2::exp crashing)
        patch_valid &= inc.array().isFinite().all();

        // avoid very large increment
        patch_valid &= inc.template lpNorm<Eigen::Infinity>() < 1e6;

        if (patch_valid) {
          transform *= SE2::exp(inc).matrix();

          const int filter_margin = 2;

          patch_valid &= img_2.InBounds(transform.translation(), filter_margin);
        }
      }
    }

    return patch_valid;
  }

  void addPoints() {
    Eigen::aligned_vector<Eigen::Vector2d> pts0;

    for (const auto& kv : transforms->observations.at(0)) {
      pts0.emplace_back(kv.second.translation().cast<double>());
    }

    KeypointsData kd;

    // num_points_cell raised 1 -> 2 -> 3: see
    // config.optical_flow_detection_grid_size's comment (vio_config.cpp)
    // for why more raw candidate corners should help tracked_ratio even
    // without improving either downstream matching path's success rate.
    // Pushed further specifically to raise how many candidates the
    // FIRST post-rotation keyframe has to work with, shrinking the
    // landmark-pool rebuild time from conversation's live measurement
    // (run 20260921_143218, ~1.8-2.2s typical).
    detectKeypoints(pyramid->at(0).lvl(0), kd,
                    config.optical_flow_detection_grid_size, 3, pts0);

    Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f> new_poses0,
        new_poses1;

    for (size_t i = 0; i < kd.corners.size(); i++) {
      Eigen::AffineCompact2f transform;
      transform.setIdentity();
      transform.translation() = kd.corners[i].cast<Scalar>();

      transforms->observations.at(0)[last_keypoint_id] = transform;
      new_poses0[last_keypoint_id] = transform;

      last_keypoint_id++;
    }

    // Diagnostic funnel for newly-detected cam0 corners specifically (not
    // points already being carried forward from earlier frames): how many
    // survive the KLT-based cam0->cam1 stereo track (before filterPoints()'s
    // epipolar check below removes more) -- see diag_new_ids_this_frame_'s
    // use in filterPoints() for the rest of the funnel.
    diag_new_ids_this_frame_.clear();
    for (const auto& kv : new_poses0) diag_new_ids_this_frame_.push_back(kv.first);
    diag_new_klt_stereo_ok_ = 0;

    if (calib.intrinsics.size() > 1) {
      trackNewPointsStereoWithDepthSeeds(pyramid->at(0), pyramid->at(1),
                                        new_poses0, new_poses1);

      for (const auto& kv : new_poses1) {
        transforms->observations.at(1).emplace(kv);
      }
      diag_new_klt_stereo_ok_ = new_poses1.size();
    }
  }

  void filterPoints() {
    if (calib.intrinsics.size() < 2) return;

    std::set<KeypointId> lm_to_remove;

    std::vector<KeypointId> kpid;
    Eigen::aligned_vector<Eigen::Vector2f> proj0, proj1;

    for (const auto& kv : transforms->observations.at(1)) {
      auto it = transforms->observations.at(0).find(kv.first);

      if (it != transforms->observations.at(0).end()) {
        proj0.emplace_back(it->second.translation());
        proj1.emplace_back(kv.second.translation());
        kpid.emplace_back(kv.first);
      }
    }

    Eigen::aligned_vector<Eigen::Vector4f> p3d0, p3d1;
    std::vector<bool> p3d0_success, p3d1_success;

    calib.intrinsics[0].unproject(proj0, p3d0, p3d0_success);
    calib.intrinsics[1].unproject(proj1, p3d1, p3d1_success);

    for (size_t i = 0; i < p3d0_success.size(); i++) {
      if (p3d0_success[i] && p3d1_success[i]) {
        const double epipolar_error =
            std::abs(p3d0[i].transpose() * E * p3d1[i]);

        if (epipolar_error > config.optical_flow_epipolar_error) {
          lm_to_remove.emplace(kpid[i]);
        }
      } else {
        lm_to_remove.emplace(kpid[i]);
      }
    }

    for (int id : lm_to_remove) {
      transforms->observations.at(1).erase(id);
    }

    // Completes the funnel started in addPoints(): of the corners newly
    // detected THIS frame, how many still have a cam1 observation after
    // this epipolar check -- distinguishes "KLT itself couldn't find a
    // stereo match" (diag_new_klt_stereo_ok_ already low) from "KLT found
    // a match but epipolar verification rejected it" (drops between
    // diag_new_klt_stereo_ok_ and this count), the two different failure
    // modes feeding the same downstream landmark-starvation symptom (see
    // [LANDMARK-DIAG] in sqrt_keypoint_vio.cpp).
    if (!diag_new_ids_this_frame_.empty()) {
      int survived = 0;
      for (KeypointId id : diag_new_ids_this_frame_) {
        if (transforms->observations.at(1).count(id)) survived++;
      }
      std::cout << "[OPTFLOW-STEREO-DIAG] new_detected="
                << diag_new_ids_this_frame_.size()
                << " klt_stereo_ok=" << diag_new_klt_stereo_ok_
                << " epipolar_survived=" << survived << std::endl;
    }
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  int64_t t_ns;

  size_t frame_counter;

  KeypointId last_keypoint_id;

  // See addPoints()/filterPoints() for how these track the newly-detected-
  // corners stereo-pairing funnel.
  std::vector<KeypointId> diag_new_ids_this_frame_;
  size_t diag_new_klt_stereo_ok_ = 0;

  VioConfig config;
  basalt::Calibration<Scalar> calib;

  OpticalFlowResult::Ptr transforms;
  // One extra frame of history behind 'transforms', kept so trackPoints()
  // can compute each track's recent 2D pixel velocity (see its own
  // comment for why) -- null until the third processed frame, since
  // there's no N-2 yet on the first tracked step.
  OpticalFlowResult::Ptr prev_transforms;
  std::shared_ptr<std::vector<basalt::ManagedImagePyr<uint16_t>>> old_pyramid,
      pyramid;

  Matrix4 E;
  // See trackNewPointsStereoWithDepthSeeds().
  Sophus::SE3<Scalar> T_c1_c0_;

  // See addIMUToQueue()/integrateRotation()'s comments. Raw IMU-frame
  // samples, timestamp-ordered; converted to a given camera's frame only
  // lazily, at integration time.
  mutable std::mutex gyro_mutex_;
  std::deque<std::pair<int64_t, Eigen::Vector3d>> gyro_queue_;
  static constexpr int64_t kGyroHistoryNs = 500'000'000;  // 500ms

  std::shared_ptr<std::thread> processing_thread;
};

}  // namespace basalt
