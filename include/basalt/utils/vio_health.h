#pragma once

#include <optional>
#include <string>

namespace basalt {

// Raw signals feeding a live VIO confidence estimate. Every field here is
// something the pipeline already computes somewhere -- this struct's only
// job is to collect them in one place; see each field's source accessor.
struct VioConfidenceInputs {
  // Fraction of the current frame's cam0 observations that matched an
  // existing landmark, in [0, 1].
  // Source: SqrtKeypointVioEstimator::getLatestTrackedRatio().
  double tracked_ratio = 1.0;

  // Stereo-triangulated point count from the most recently processed
  // keyframe, or nullopt if no loop-closure module is running at all.
  // Source: OnlineLoopClosure::getLatestTriangulatedPoints().
  std::optional<int> triangulated_points;

  // Set once the core estimator's optimization hits a degenerate/
  // numerically-invalid linearization -- trumps every other input below,
  // since a degraded optimizer state makes every other signal suspect.
  // Source: SqrtKeypointVioEstimator::vio_health.degraded.
  bool numerically_degraded = false;

  // Rotation rate at the timestamp of the most recently consumed IMU
  // sample, rad/s. Informational only in this version: there is not yet
  // real scenario data (fast-rotation characterization, still pending)
  // to calibrate a "confidence drops above rate X" threshold against, so
  // it is carried through computeVioConfidence()'s output for visibility
  // but does not currently affect the score.
  // Source: SqrtKeypointVioEstimator::getLatestGyroNorm().
  double gyro_norm = 0.0;
};

struct VioConfidenceConfig {
  // Deliberately the SAME threshold VioConfig::vio_new_kf_keypoints_thresh
  // already uses for keyframe-insertion decisions (default 0.7f) -- not
  // an independently re-tuned number. Below this, the estimator is
  // already deciding "not enough of this frame is trustworthy," which is
  // exactly what a tracking-quality signal wants to know too.
  double min_tracked_ratio = 0.7;

  // Mirrors OnlineLoopClosure's kMinTriangulatedPointsForDatabase (25):
  // below this, a keyframe isn't even good enough to serve as a future
  // loop-closure match target, so treat it as a real quality concern for
  // VIO's own confidence, not just a loop-closure-specific gate.
  int min_triangulated_points = 25;
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
