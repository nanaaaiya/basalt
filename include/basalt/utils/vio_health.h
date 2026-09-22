#pragma once

#include <optional>
#include <string>

namespace basalt {

// Raw signals feeding a live VIO confidence estimate. Every field here is
// something the pipeline already computes somewhere -- this struct's only
// job is to collect them in one place; see each field's source accessor.
struct VioConfidenceInputs {
  // Absolute count of the current frame's cam0 observations that matched
  // an existing landmark. Deliberately NOT a ratio (tracked_count /
  // total_observed_count): real Pi5/OAK-D Lite runs showed tracked_ratio
  // chronically sitting in the 0.0-0.2 range even on well-tracked frames
  // (see VioConfig::vio_new_kf_keypoints_thresh's comment in
  // vio_config.cpp, and the starvation-gate history in
  // online_loop_closure.cpp), so a ratio-based threshold here saturated
  // this signal near its floor for nearly an entire live test (run
  // pi5-63f519dd, 2026-09-22: confidence stuck at 0.10-0.20 for the full
  // ~111s run, including well-tracked stretches) instead of discriminating
  // good stretches from bad ones.
  // Source: SqrtKeypointVioEstimator::getLatestTrackedCount().
  int tracked_count = 999;

  // Stereo-triangulated point count from the most recently processed
  // keyframe, or nullopt if no loop-closure module is running at all.
  // Source: OnlineLoopClosure::getLatestTriangulatedPoints().
  std::optional<int> triangulated_points;

  // Set once the core estimator's optimization hits a degenerate/
  // numerically-invalid linearization -- trumps every other input below,
  // since a degraded optimizer state makes every other signal suspect.
  // Source: SqrtKeypointVioEstimator::vio_health.degraded.
  bool numerically_degraded = false;

  // True for a short cooldown after the loop-closure drift gate's
  // max-hold timeout gave up waiting and accepted an unconfirmed
  // correction (as opposed to 3 consecutive solves actually confirming
  // it) -- see DriftGateEvent::kReleasedForced and
  // OnlineLoopClosure::isRecentlyForceReleased(). The corrected pose
  // right after this is one the system itself never verified, which is
  // exactly what a confidence consumer needs to know, separately from
  // whatever tracked_ratio happens to read at that instant.
  bool recently_forced_drift_release = false;

  // Rotation rate at the timestamp of the most recently consumed IMU
  // sample, rad/s. Informational only in this version: there is not yet
  // real scenario data (fast-rotation characterization, still pending)
  // to calibrate a "confidence drops above rate X" threshold against, so
  // it is carried through computeVioConfidence()'s output for visibility
  // but does not currently affect the score.
  // Source: SqrtKeypointVioEstimator::getLatestGyroNorm().
  double gyro_norm = 0.0;

  // True once the joint (vision+IMU) optimized pose has disagreed with
  // pure IMU-integration's prediction for several consecutive frames --
  // the signature of a dynamic-scene violation (flowing water, a close
  // moving object) large enough to fool per-point robust loss, which
  // only helps when bad points are a minority.
  // Source: SqrtKeypointVioEstimator::isImuVisionDisagreement().
  bool imu_vision_disagreement = false;
};

struct VioConfidenceConfig {
  // Was a ratio threshold mirroring VioConfig::vio_new_kf_keypoints_thresh
  // -- but that mirror had already gone stale (that config field was
  // separately lowered 0.7 -> 0.3 once real hardware showed tracked_ratio
  // chronically sitting in 0.0-0.2, this comment's old value was never
  // updated to match), and even 0.3 wouldn't have fixed it: this rig's
  // real tracked_ratio *mean* sits below that too. Switched to the same
  // absolute-count threshold already validated for the same purpose on
  // the same rig -- see kBiasFreezeTrackedCountThresh in
  // sqrt_keypoint_vio.h and the starvation-gate history in
  // online_loop_closure.cpp (also originally ratio-based, also switched
  // to absolute count after density increases diluted the ratio).
  int min_tracked_count = 8;

  // Mirrors OnlineLoopClosure's kMinTriangulatedPointsForDatabase (20,
  // lowered from 25 after real low-texture/distant-background sessions
  // showed triangulated points never clearing 25 at all -- see that
  // constant's own history comment in online_loop_closure.cpp): below
  // this, a keyframe isn't even good enough to serve as a future
  // loop-closure match target, so treat it as a real quality concern for
  // VIO's own confidence, not just a loop-closure-specific gate.
  int min_triangulated_points = 20;
};

struct VioConfidence {
  // A coarse, interpretable score in [0, 1] -- NOT a calibrated
  // probability. This is a threshold cascade, not a learned/weighted
  // blend, by deliberate choice: it starts from thresholds already
  // relied on elsewhere in this codebase (see VioConfidenceConfig),
  // and is expected to be retuned once real scenario-characterization
  // data exists (fast rotation / low texture / low light / fast
  // translation) rather than treated as finished on first use.
  double score = 1.0;

  // Which single check most limited the score above; "nominal" if
  // nothing did. Always exactly one reason, even if multiple inputs
  // are simultaneously weak -- report the most severe one, since a
  // consumer deciding whether to trust VIO needs one clear answer, not
  // a list to interpret itself.
  std::string primary_reason = "nominal";
};

// Pure function, no owned state -- callable identically from the live
// app (oak_d_vio.cpp) and any offline scenario-report generator, and
// trivially unit-testable against synthetic inputs.
VioConfidence computeVioConfidence(const VioConfidenceInputs& in,
                                   const VioConfidenceConfig& cfg = {});

}  // namespace basalt
