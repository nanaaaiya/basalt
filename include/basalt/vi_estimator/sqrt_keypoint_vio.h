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

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>

#include <basalt/imu/preintegration.h>
#include <basalt/utils/time_utils.hpp>

#include <basalt/vi_estimator/sqrt_ba_base.h>
#include <basalt/vi_estimator/vio_estimator.h>

#include <basalt/imu/preintegration.h>

namespace basalt {

template <class Scalar_>
class SqrtKeypointVioEstimator : public VioEstimatorBase,
                                 public SqrtBundleAdjustmentBase<Scalar_> {
 public:
  using Scalar = Scalar_;

  typedef std::shared_ptr<SqrtKeypointVioEstimator> Ptr;

  static const int N = 9;
  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
  using Vec4 = Eigen::Matrix<Scalar, 4, 1>;
  using VecN = Eigen::Matrix<Scalar, N, 1>;
  using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using MatN3 = Eigen::Matrix<Scalar, N, 3>;
  using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using SE3 = Sophus::SE3<Scalar>;

  using typename SqrtBundleAdjustmentBase<Scalar>::RelLinData;
  using typename SqrtBundleAdjustmentBase<Scalar>::AbsLinData;

  using BundleAdjustmentBase<Scalar>::computeError;
  using BundleAdjustmentBase<Scalar>::get_current_points;
  using BundleAdjustmentBase<Scalar>::computeDelta;
  using BundleAdjustmentBase<Scalar>::computeProjections;
  using BundleAdjustmentBase<Scalar>::triangulate;
  using BundleAdjustmentBase<Scalar>::backup;
  using BundleAdjustmentBase<Scalar>::restore;
  using BundleAdjustmentBase<Scalar>::getPoseStateWithLin;
  using BundleAdjustmentBase<Scalar>::computeModelCostChange;

  using SqrtBundleAdjustmentBase<Scalar>::linearizeHelper;
  using SqrtBundleAdjustmentBase<Scalar>::linearizeAbsHelper;
  using SqrtBundleAdjustmentBase<Scalar>::linearizeRel;
  using SqrtBundleAdjustmentBase<Scalar>::linearizeAbs;
  using SqrtBundleAdjustmentBase<Scalar>::updatePoints;
  using SqrtBundleAdjustmentBase<Scalar>::updatePointsAbs;
  using SqrtBundleAdjustmentBase<Scalar>::linearizeMargPrior;
  using SqrtBundleAdjustmentBase<Scalar>::computeMargPriorError;
  using SqrtBundleAdjustmentBase<Scalar>::computeMargPriorModelCostChange;
  using SqrtBundleAdjustmentBase<Scalar>::checkNullspace;
  using SqrtBundleAdjustmentBase<Scalar>::checkEigenvalues;

  SqrtKeypointVioEstimator(const Eigen::Vector3d& g,
                           const basalt::Calibration<double>& calib,
                           const VioConfig& config);

  void initialize(int64_t t_ns, const Sophus::SE3d& T_w_i,
                  const Eigen::Vector3d& vel_w_i, const Eigen::Vector3d& bg,
                  const Eigen::Vector3d& ba) override;

  void initialize(const Eigen::Vector3d& bg,
                  const Eigen::Vector3d& ba) override;

  virtual ~SqrtKeypointVioEstimator() { maybe_join(); }

  inline void maybe_join() override {
    if (processing_thread) {
      processing_thread->join();
      processing_thread.reset();
    }
  }

  void addIMUToQueue(const ImuData<double>::Ptr& data) override;
  void addVisionToQueue(const OpticalFlowResult::Ptr& data) override;

  typename ImuData<Scalar>::Ptr popFromImuDataQueue();

  bool measure(const OpticalFlowResult::Ptr& opt_flow_meas,
               const typename IntegratedImuMeasurement<Scalar>::Ptr& meas);

  // int64_t propagate();
  // void addNewState(int64_t data_t_ns);

  void optimize_and_marg(const std::map<int64_t, int>& num_points_connected,
                         const std::unordered_set<KeypointId>& lost_landmaks);

  void marginalize(const std::map<int64_t, int>& num_points_connected,
                   const std::unordered_set<KeypointId>& lost_landmaks);
  void optimize();

  void debug_finalize() override;

  void logMargNullspace();
  Eigen::VectorXd checkMargNullspace() const;
  Eigen::VectorXd checkMargEigenvalues() const;

  int64_t get_t_ns() const {
    return frame_states.at(last_state_t_ns).getState().t_ns;
  }
  const SE3& get_T_w_i() const {
    return frame_states.at(last_state_t_ns).getState().T_w_i;
  }
  const Vec3& get_vel_w_i() const {
    return frame_states.at(last_state_t_ns).getState().vel_w_i;
  }

  const PoseVelBiasState<Scalar>& get_state() const {
    return frame_states.at(last_state_t_ns).getState();
  }
  PoseVelBiasState<Scalar> get_state(int64_t t_ns) const {
    PoseVelBiasState<Scalar> state;

    auto it = frame_states.find(t_ns);

    if (it != frame_states.end()) {
      return it->second.getState();
    }

    auto it2 = frame_poses.find(t_ns);
    if (it2 != frame_poses.end()) {
      state.T_w_i = it2->second.getPose();
    }

    return state;
  }

  void setMaxStates(size_t val) override { max_states = val; }
  void setMaxKfs(size_t val) override { max_kfs = val; }

  Eigen::aligned_vector<SE3> getFrameStates() const {
    Eigen::aligned_vector<SE3> res;

    for (const auto& kv : frame_states) {
      res.push_back(kv.second.getState().T_w_i);
    }

    return res;
  }

  Eigen::aligned_vector<SE3> getFramePoses() const {
    Eigen::aligned_vector<SE3> res;

    for (const auto& kv : frame_poses) {
      res.push_back(kv.second.getPose());
    }

    return res;
  }

  Eigen::aligned_map<int64_t, SE3> getAllPosesMap() const {
    Eigen::aligned_map<int64_t, SE3> res;

    for (const auto& kv : frame_poses) {
      res[kv.first] = kv.second.getPose();
    }

    for (const auto& kv : frame_states) {
      res[kv.first] = kv.second.getState().T_w_i;
    }

    return res;
  }

  Sophus::SE3d getT_w_i_init() override {
    return T_w_i_init.template cast<double>();
  }

  // Set (not aborted) when optimize()'s linearization hits a degenerate/
  // numerically-invalid case -- see optimize() in the .cpp for the failure
  // this replaces (a hard abort()). No consumer reads this yet in this
  // repo; it exists so the live app / a future fusion layer can decide
  // whether to trust the published pose, since this class can no longer
  // guarantee that on its own by crashing instead.
  struct VioHealth {
    std::atomic<bool> degraded{false};
    std::atomic<int64_t> last_degraded_t_ns{-1};
    std::atomic<int> consecutive_degraded_count{0};
  };
  VioHealth vio_health;

  // Rotation rate at the timestamp of the most recent IMU sample VIO has
  // actually consumed (see popFromImuDataQueue() in the .cpp) -- "how
  // fast is this rotating right now," for a confidence signal or
  // scenario-characterization tooling. gyro is bias-corrected by the
  // time it reaches here (see the .cpp call sites).
  // Fraction of the current frame's cam0 observations that matched an
  // existing landmark (vs. never-before-seen) -- the same ratio
  // vio_new_kf_keypoints_thresh already gates keyframe insertion on, now
  // also exposed as a standalone tracking-quality signal.
  double getLatestTrackedRatio() const override { return latest_tracked_ratio; }

  // Raw counts behind getLatestTrackedRatio() -- added to distinguish "few
  // features exist at all this frame" (a detection/texture/exposure
  // problem) from "plenty exist but few matched an existing landmark" (a
  // frame-to-frame tracking/motion-blur problem), which the ratio alone
  // can't tell apart. tracked_count = connected0, total_observed_count =
  // connected0 + unconnected_obs0.size() (see measure() in the .cpp).
  int getLatestTrackedCount() const override { return latest_tracked_count; }
  int getLatestTotalObservedCount() const override {
    return latest_total_observed_count;
  }

  double getLatestGyroNorm() const override { return latest_gyro_norm; }
  double getLatestAccelNorm() const override { return latest_accel_norm; }
  Eigen::Vector3d getLatestGyro() const {
    std::lock_guard<std::mutex> lock(latest_gyro_mutex);
    return latest_gyro;
  }

  // The optimizer's current estimate of the accel/gyro sensor bias --
  // previously invisible outside the estimator, so a runaway raw
  // trajectory could only be diagnosed after the fact from the position
  // blowup, never by watching the bias state itself as it happens. Read
  // from the same state the sliding window already maintains (see
  // measure()'s next_state), not recomputed.
  Eigen::Vector3d getLatestAccelBias() const override {
    std::lock_guard<std::mutex> lock(latest_bias_mutex);
    return latest_accel_bias;
  }
  Eigen::Vector3d getLatestGyroBias() const override {
    std::lock_guard<std::mutex> lock(latest_bias_mutex);
    return latest_gyro_bias;
  }

  bool isDegraded() const override { return vio_health.degraded; }

  // Detect-and-flag defense against dynamic-scene corruption (a hand
  // waved close to a stationary camera, flowing water filling a large
  // fraction of the frame, etc.) that per-point robust loss can't catch
  // once the corrupted points are the majority, not a minority. Compares
  // the joint (vision+IMU) optimized pose against what pure IMU
  // integration alone predicted for the same step (see measure()) --
  // vision pulling the pose away from that prediction is normal noise on
  // any single frame, but if it persists over many consecutive frames,
  // that's the same signature a genuinely static-world-violating scene
  // produces, independent of what fraction of tracked points are
  // affected (unlike tracked_ratio/Huber loss, which only help when bad
  // points are a minority).
  bool isImuVisionDisagreement() const override {
    return imu_vision_disagreement;
  }
  double getLatestImuVisionDisagreementM() const override {
    return latest_imu_vision_disagreement_m;
  }

  // Whether vision's residual weight is currently reduced in response to
  // isImuVisionDisagreement() -- see the reweighting members' comment for
  // the full design (bounded to the flagged window, hard time cap).
  // False both when nominal (flag clear) AND when the time cap has been
  // exceeded (flag still set but reweighting gave up for this episode) --
  // getLatestImuVisionDisagreementM() distinguishes those two if needed.
  bool isImuVisionReweightActive() const override {
    return imu_vision_reweight_active_;
  }

  // See kBiasFreezeTrackedCountThresh's comment. Same true/false
  // semantics as isImuVisionReweightActive() above (false both when
  // nominal and when the time cap gave up for this episode).
  bool isBiasFreezeActive() const override { return bias_freeze_active_; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  using BundleAdjustmentBase<Scalar>::frame_poses;
  using BundleAdjustmentBase<Scalar>::frame_states;
  using BundleAdjustmentBase<Scalar>::lmdb;
  using BundleAdjustmentBase<Scalar>::obs_std_dev;
  using BundleAdjustmentBase<Scalar>::huber_thresh;
  using BundleAdjustmentBase<Scalar>::calib;

 private:
  bool take_kf;
  int frames_after_kf;
  std::set<int64_t> kf_ids;

  int64_t last_state_t_ns;

  std::atomic<double> latest_tracked_ratio{1.0};
  std::atomic<int> latest_tracked_count{0};
  std::atomic<int> latest_total_observed_count{0};
  std::atomic<double> latest_gyro_norm{0.0};
  // Raw accelerometer magnitude (specific force, not yet gravity- or
  // bias-corrected) at the timestamp of the most recently consumed IMU
  // sample -- the linear-motion counterpart to latest_gyro_norm, added
  // for the disagreement detector's acceleration gate (see
  // kImuVisionDisagreementMaxExcessAccelMps2 below).
  std::atomic<double> latest_accel_norm{0.0};
  mutable std::mutex latest_gyro_mutex;
  Eigen::Vector3d latest_gyro{Eigen::Vector3d::Zero()};

  mutable std::mutex latest_bias_mutex;
  Eigen::Vector3d latest_accel_bias{Eigen::Vector3d::Zero()};
  Eigen::Vector3d latest_gyro_bias{Eigen::Vector3d::Zero()};
  Eigen::aligned_map<int64_t, IntegratedImuMeasurement<Scalar>> imu_meas;

  // IMU-vision disagreement detector (see isImuVisionDisagreement() above).
  // History: 0.05m thresh / 10-frame persistence (initial guesses) -> 0.02m
  // / 3 frames, retuned against a real captured hand-wave
  // (run_logs/20260918_142001): the actual disagreement bump was real and
  // clearly hand-wave-shaped, but only ~6-7 frames wide (~0.4-0.5s) and
  // only cleared 0.05m on a single frame, so the original 10-frame
  // requirement -- sized for a SUSTAINED violation like flowing water --
  // never came close to firing for what was actually a brief, reflexive
  // wave (the original motivating case for this whole detector). Across
  // that same run's 802 frames (hand-wave included), p99 stayed at 0.03m,
  // so 0.02m/3 frames should catch this class of event without much
  // false-positive risk from ordinary noise -- still unvalidated against
  // a live re-test, expect further tuning.
  // Threshold: how far (m) the joint optimizer is allowed to pull a
  // single frame's position away from the pure-IMU prediction before
  // counting it as "disagreeing" -- predictState() already accounts for
  // the current velocity estimate, so ordinary continued motion doesn't
  // by itself produce a large gap here; this is meant to catch NEW,
  // unexplained corrections, not motion in general.
  static constexpr double kImuVisionDisagreementThreshM = 0.02;
  // Persistence: consecutive frames the disagreement must stay above
  // threshold before flagging -- a single frame is expected sensor noise
  // or the normal transient stress of a real aggressive maneuver (camera-
  // IMU time sync/extrinsics are never perfect); still requiring a few
  // consecutive frames (not just one) keeps a single noisy sample from
  // triggering it.
  static constexpr int kImuVisionDisagreementPersistenceFrames = 3;
  // Rotation-rate gate, added after the 0.02m/3-frame retune above turned
  // out to false-positive heavily during genuine aggressive motion (two
  // separate real test flights: 41 fires/11 episodes and 74 fires/14
  // episodes over ~26-40s each, both with gyro_norm means around 2 rad/s
  // and peaks past 20 rad/s -- vs. a real stationary dynamic-scene
  // corruption event, which measured gyro_norm mean 0.009, max 0.28
  // rad/s). Real fast rotation stresses camera-IMU time sync/extrinsics
  // enough to produce the same kind of brief position disagreement a
  // genuine corruption does, so disagreement alone can't tell the two
  // apart -- gyro_norm can. Threshold sits with wide margin above the
  // stationary-corruption case's observed max (0.28) and well below the
  // aggressive-motion cases' mean (~2), so legitimate maneuvering is
  // excluded without narrowing what counts as "the camera is basically
  // not rotating" for the corruption case itself.
  static constexpr double kImuVisionDisagreementMaxGyroNormRadS = 0.5;
  // Linear-acceleration counterpart to the gyro gate above -- added after
  // the gyro gate alone only partly fixed a second real aggressive-motion
  // test (74 fires/14 episodes -> 58/9, vs. 41/11 -> 5/1 for the first):
  // checking gyro_norm at each of that test's remaining false triggers
  // showed it was actually LOW there (0.05-0.27 rad/s), meaning that
  // test's motion was translation-heavy rather than purely rotational --
  // a rotation-rate gate alone can't see that. Compares the raw
  // accelerometer magnitude (specific force, not yet bias-corrected --
  // fine for a coarse gate, the bias is ~0.03 m/s^2 per the [VIO-BIAS]
  // telemetry, negligible next to what this is meant to catch) against
  // gravity's magnitude; a large deviation means real linear acceleration
  // is happening, which -- like fast rotation -- can stress camera-IMU
  // calibration enough to produce the same kind of disagreement a genuine
  // corruption does. Starting value, NOT validated the way the gyro
  // threshold was: retroactively replayed against four real post-
  // deployment test captures (2307 frames total) once accel telemetry
  // existed, and the pattern this gate targets (low rotation, high
  // linear acceleration, high disagreement) occurred ZERO times. What
  // looked like a translation-caused false-positive residual in the
  // second aggressive test instead correlated with erratic tracked_count
  // (0, 0, 10, 0, 0, 0, 0, 30, 0, 19 -- landmark-database instability
  // after a violent disruption, not instantaneous kinematics) -- a
  // mechanism this gate was never meant to address and arguably
  // shouldn't suppress anyway, since the pose really is unreliable then.
  // Net effect so far: a harmless, currently-inert safeguard rather than
  // a confirmed fix. Leave as-is unless/until a real translation-heavy,
  // low-rotation disturbance is actually captured to test it against.
  static constexpr double kImuVisionDisagreementMaxExcessAccelMps2 = 3.0;
  int imu_vision_disagreement_count_ = 0;
  std::atomic<bool> imu_vision_disagreement{false};
  std::atomic<double> latest_imu_vision_disagreement_m{0.0};

  // Detect-AND-REWEIGHT: while imu_vision_disagreement is flagged, vision's
  // residual weight in the optimizer is temporarily reduced (obs_std_dev
  // raised, so vision counts for less against the IMU factor) instead of
  // just reporting the flag. Deliberately bounded on two axes, both
  // learned the hard way earlier this session: (1) a global, permanent
  // version of "trust IMU more" (halved accel_bias_std/gyro_noise_std in
  // the calibration file) caused 225-300m raw-trajectory runaways,
  // because it also suppressed the bias self-correction vision provides
  // during the many frames where nothing is actually wrong -- so this
  // ONLY applies while the flag is active, snapping back to nominal the
  // instant it clears; (2) even scoped to the flagged window, a genuinely
  // sustained event (flowing water) must not let reweighting run
  // indefinitely, since that reproduces the same bias-runs-free failure
  // just over a shorter, still-real window -- kImuVisionReweightMaxDurationS
  // forces a return to nominal trust after that long, accepting the
  // (smaller) risk of trusting a still-flagged vision stream over the
  // (larger, proven) risk of unbounded uncorrected IMU drift.
  //
  // One-frame lag by construction: a given frame's disagreement can only
  // be computed AFTER that frame's optimize_and_marg() already ran (it
  // compares the optimized result against the IMU prediction), so
  // reweighting for frame N is decided from frame N-1's flag, applied
  // before optimize_and_marg() runs, matching frame_states.at(...) reads
  // throughout this file (see measure()). At ~15-20Hz this is single-
  // digit milliseconds of lag, not a design compromise worth avoiding at
  // the cost of running optimize() twice per frame.
  //
  // Both constants are unvalidated starting guesses (no live reweighting
  // test has run yet) -- kImuVisionReweightFactor is a coarse, whole-
  // frame downweight (this detector only knows "something disagrees
  // overall", not which specific points are bad, so it can't do a
  // targeted per-point rejection instead). Expect to retune once real
  // before/after data exists on both a disturbance run and a clean run.
  static constexpr double kImuVisionReweightFactor = 3.0;
  static constexpr double kImuVisionReweightMaxDurationS = 5.0;
  std::atomic<bool> imu_vision_reweight_active_{false};
  std::chrono::steady_clock::time_point imu_vision_reweight_start_wall_;
  Scalar nominal_obs_std_dev{0};

  // Bias freeze: a SEPARATE mechanism from the reweight above, for a
  // different failure mode found on a real walking test (triangle-
  // pattern accuracy test, Pi5, 2026-09-21) -- during a sustained
  // near-zero-tracked_count stretch (camera rotating through a corner,
  // no vision correction at all, not just weaker vision), the bias
  // states are only constrained by IMU self-consistency and their own
  // process-noise prior (accel_bias_sqrt_weight/gyro_bias_sqrt_weight,
  // derived from the calibration's accel_bias_std/gyro_bias_std). Two
  // real captures showed accel_bias_norm jumping from a healthy ~0.1 to
  // 1.0-2.4 m/s^2 within a handful of seconds during exactly this kind
  // of window, then (after vision returned and normal operation
  // resumed for tens of seconds) the raw trajectory diverging
  // exponentially to 1700-2400m -- a genuinely different mechanism from
  // the [ONLINE-LOOP] drift gate/starvation trigger built earlier the
  // same day, which only protects the PUBLISHED/corrected pose, not
  // this estimator's own internal state.
  //
  // The fix is the mirror image of the reweight mechanism above: instead
  // of trusting vision more, temporarily make the bias states harder to
  // move (raise their process-noise weight) while there's nothing to
  // validate a bias change against. This carries the EXACT SAME risk
  // already learned the hard way earlier this session (see
  // kImuVisionReweightFactor's comment): a bias made too rigid for too
  // long stops self-correcting even during the many frames where
  // nothing is actually wrong, which is what caused the original
  // 225-300m runaway from halving accel_bias_std globally and
  // permanently in the calibration file. Given how often tracked_count
  // dips on this rig (see conversation -- chronic, not rare), an
  // unbounded or over-eager freeze here would reproduce that exact
  // failure. Same two-axis bound as the reweight mechanism: (1) only
  // active while latest_tracked_count has been continuously below
  // kBiasFreezeTrackedCountThresh for at least
  // kBiasFreezeMinPersistenceS (not on a single noisy low reading), and
  // resets once a recovery has ITSELF persisted for
  // kBiasFreezeRecoveryGraceS -- NOT the instant tracked_count recovers,
  // which is what this originally did until a real live test
  // (2026-09-22, untextured wall) showed the analogous drift-gate
  // starvation trigger (see kStarvationRecoveryGraceS in
  // online_loop_closure.cpp) missing an entire ~13s bad stretch because
  // sub-second good flickers kept discarding its accumulated bad-streak
  // time; this mechanism has the identical instant-reset shape, so it
  // gets the identical fix; (2) even within a flagged episode,
  // kBiasFreezeMaxDurationS forces a return to nominal after that long,
  // accepting a bounded window of uncorrected bias over an unbounded
  // one. Does not claim to fully prevent the runaway (the observed
  // captures ran for many seconds of near-zero tracked_count, likely
  // still exceeding this cap in the worst case) -- it bounds how far the
  // bias can move DURING the blackout, which is a real reduction versus
  // today's unconstrained behavior, not a proven complete fix. Needs a
  // live retest against the same triangle-walk scenario before being
  // trusted further.
  static constexpr int kBiasFreezeTrackedCountThresh = 8;
  static constexpr double kBiasFreezeMinPersistenceS = 1.0;
  static constexpr double kBiasFreezeWeightMultiplier = 20.0;
  static constexpr double kBiasFreezeMaxDurationS = 5.0;
  // Roughly half kBiasFreezeMinPersistenceS, same ratio as
  // kStarvationRecoveryGraceS/kStarvationPersistenceSeconds -- not
  // independently tuned.
  static constexpr double kBiasFreezeRecoveryGraceS = 0.5;
  bool low_tracked_count_streak_active_ = false;
  std::chrono::steady_clock::time_point low_tracked_count_since_wall_;
  bool has_good_tracked_count_streak_ = false;
  std::chrono::steady_clock::time_point good_tracked_count_streak_since_wall_;
  std::atomic<bool> bias_freeze_active_{false};
  std::chrono::steady_clock::time_point bias_freeze_start_wall_;
  Vec3 nominal_accel_bias_sqrt_weight, nominal_gyro_bias_sqrt_weight;

  // Position-propagation suppression -- see its use in initialize()'s
  // preintegration loop for the full reasoning (complements the bias
  // freeze above: that keeps the bias STATE from wandering during
  // starvation, this keeps THIS FRAME's accel samples from being
  // integrated at full trust into the preintegrated factor's
  // relative-position/velocity terms, the quadratic-in-time error
  // source behind every raw-trajectory runaway measured on the
  // 2026-09-21 triangle-walk tests). Reuses kBiasFreezeTrackedCountThresh
  // as the same "starved" signal rather than a separate threshold --
  // both mechanisms react to the identical condition, just at different
  // points in the pipeline. No persistence delay or max-duration cap
  // unlike the bias freeze: this only loosens one interval's
  // MEASUREMENT trust, not a standing prior, so it's safe to react
  // every frame and never risks suppressing future correction. Value is
  // an unvalidated starting guess -- large enough that the optimizer
  // effectively ignores this interval's accel-derived position/velocity
  // contribution and leans on the marginalization prior / other edges
  // instead, without being literally infinite.
  static constexpr double kStarvedAccelCovInflationFactor = 1e4;

  const Vec3 g;

  // Input

  Eigen::aligned_map<int64_t, OpticalFlowResult::Ptr> prev_opt_flow_res;

  std::map<int64_t, int> num_points_kf;

  // Marginalization
  MargLinData<Scalar> marg_data;

  // Used only for debug and log purporses.
  MargLinData<Scalar> nullspace_marg_data;

  Vec3 gyro_bias_sqrt_weight, accel_bias_sqrt_weight;

  size_t max_states;
  size_t max_kfs;

  SE3 T_w_i_init;

  bool initialized;
  bool opt_started;

  VioConfig config;

  constexpr static Scalar vee_factor = Scalar(2.0);
  constexpr static Scalar initial_vee = Scalar(2.0);
  Scalar lambda, min_lambda, max_lambda, lambda_vee;

  std::shared_ptr<std::thread> processing_thread;

  // timing and stats
  ExecutionStats stats_all_;
  ExecutionStats stats_sums_;
};
}  // namespace basalt
