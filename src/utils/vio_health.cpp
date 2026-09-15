#include <basalt/utils/vio_health.h>

namespace basalt {

VioConfidence computeVioConfidence(const VioConfidenceInputs& in,
                                   const VioConfidenceConfig& cfg) {
  // Ordered most-severe-first: the first check that fires wins, since a
  // consumer needs one clear answer, not every simultaneously-weak input.
  if (in.numerically_degraded) {
    return {0.0, "numerically_degraded"};
  }

  if (in.tracked_ratio < cfg.min_tracked_ratio) {
    return {0.2, "low_tracked_keypoint_ratio"};
  }

  if (in.triangulated_points.has_value() &&
      *in.triangulated_points < cfg.min_triangulated_points) {
    return {0.4, "low_triangulation_yield"};
  }

  return {1.0, "nominal"};
}

}  // namespace basalt
