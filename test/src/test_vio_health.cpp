#include <basalt/utils/vio_health.h>

#include "gtest/gtest.h"

TEST(VioHealthTestSuite, NominalInputsScoreHigh) {
  basalt::VioConfidenceInputs in;
  in.tracked_ratio = 0.95;
  in.triangulated_points = 40;
  in.numerically_degraded = false;
  in.gyro_norm = 0.05;

  auto out = basalt::computeVioConfidence(in);

  EXPECT_DOUBLE_EQ(out.score, 1.0);
  EXPECT_EQ(out.primary_reason, "nominal");
}

TEST(VioHealthTestSuite, MissingTriangulatedPointsIsIgnoredNotPenalized) {
  // No loop-closure module running -> nullopt. Should not be treated as
  // "zero points" (which would wrongly penalize a pure-VIO-only setup).
  basalt::VioConfidenceInputs in;
  in.tracked_ratio = 0.95;
  in.triangulated_points = std::nullopt;

  auto out = basalt::computeVioConfidence(in);

  EXPECT_DOUBLE_EQ(out.score, 1.0);
  EXPECT_EQ(out.primary_reason, "nominal");
}

TEST(VioHealthTestSuite, NumericalDegradationTrumpsEverythingElse) {
  basalt::VioConfidenceInputs in;
  in.tracked_ratio = 0.99;    // otherwise perfect
  in.triangulated_points = 100;
  in.numerically_degraded = true;

  auto out = basalt::computeVioConfidence(in);

  EXPECT_DOUBLE_EQ(out.score, 0.0);
  EXPECT_EQ(out.primary_reason, "numerically_degraded");
}

TEST(VioHealthTestSuite, LowTrackedRatioReportedBeforeLowTriangulation) {
  // Both inputs are weak -- the most severe single reason should win,
  // not an average of the two.
  basalt::VioConfidenceInputs in;
  in.tracked_ratio = 0.3;         // below default 0.7 threshold
  in.triangulated_points = 5;     // also below default 20 threshold
  in.numerically_degraded = false;

  auto out = basalt::computeVioConfidence(in);

  EXPECT_EQ(out.primary_reason, "low_tracked_keypoint_ratio");
}

TEST(VioHealthTestSuite, LowTriangulationYieldAloneIsReported) {
  basalt::VioConfidenceInputs in;
  in.tracked_ratio = 0.95;        // fine on its own
  in.triangulated_points = 8;     // below default 20 threshold -- matches
                                  // the stereo-yield-collapse bug this
                                  // signal is meant to help surface
  in.numerically_degraded = false;

  auto out = basalt::computeVioConfidence(in);

  EXPECT_EQ(out.primary_reason, "low_triangulation_yield");
  EXPECT_LT(out.score, 1.0);
}

TEST(VioHealthTestSuite, ThresholdsAreConfigurableNotHardcoded) {
  basalt::VioConfidenceInputs in;
  in.tracked_ratio = 0.5;  // below the default 0.7, but...
  in.triangulated_points = 25;

  basalt::VioConfidenceConfig cfg;
  cfg.min_tracked_ratio = 0.4;  // ...above this looser threshold
  cfg.min_triangulated_points = 25;

  auto out = basalt::computeVioConfidence(in, cfg);

  EXPECT_EQ(out.primary_reason, "nominal");
}

TEST(VioHealthTestSuite, RecentForcedDriftReleaseReportedEvenIfOtherwiseNominal) {
  basalt::VioConfidenceInputs in;
  in.tracked_ratio = 0.95;       // otherwise perfect
  in.triangulated_points = 40;
  in.recently_forced_drift_release = true;

  auto out = basalt::computeVioConfidence(in);

  EXPECT_EQ(out.primary_reason, "recent_forced_drift_release");
  EXPECT_LT(out.score, 1.0);
}

TEST(VioHealthTestSuite, RecentForcedDriftReleaseReportedBeforeLowTrackedRatio) {
  // Both weak -- the more specific, time-bounded signal (we KNOW this
  // exact pose was never confirmed) should win over the inferred one.
  basalt::VioConfidenceInputs in;
  in.tracked_ratio = 0.3;  // also below default 0.7 threshold
  in.recently_forced_drift_release = true;

  auto out = basalt::computeVioConfidence(in);

  EXPECT_EQ(out.primary_reason, "recent_forced_drift_release");
}

TEST(VioHealthTestSuite, NumericalDegradationTrumpsForcedDriftReleaseToo) {
  basalt::VioConfidenceInputs in;
  in.tracked_ratio = 0.99;
  in.recently_forced_drift_release = true;
  in.numerically_degraded = true;

  auto out = basalt::computeVioConfidence(in);

  EXPECT_DOUBLE_EQ(out.score, 0.0);
  EXPECT_EQ(out.primary_reason, "numerically_degraded");
}

TEST(VioHealthTestSuite, ImuVisionDisagreementReportedEvenIfOtherwiseNominal) {
  basalt::VioConfidenceInputs in;
  in.tracked_ratio = 0.95;  // otherwise perfect
  in.triangulated_points = 40;
  in.imu_vision_disagreement = true;

  auto out = basalt::computeVioConfidence(in);

  EXPECT_EQ(out.primary_reason, "imu_vision_disagreement");
  EXPECT_LT(out.score, 1.0);
}

TEST(VioHealthTestSuite, ImuVisionDisagreementReportedBeforeLowTrackedRatio) {
  // Both weak -- the dynamic-scene-violation signal should win over the
  // generic tracking-quality one, since it's the more specific/actionable
  // explanation (and tracked_ratio alone can look fine even when a large,
  // coherently-moving object like water dominates the tracked points).
  basalt::VioConfidenceInputs in;
  in.tracked_ratio = 0.3;  // also below default 0.7 threshold
  in.imu_vision_disagreement = true;

  auto out = basalt::computeVioConfidence(in);

  EXPECT_EQ(out.primary_reason, "imu_vision_disagreement");
}

TEST(VioHealthTestSuite, NumericalDegradationTrumpsImuVisionDisagreementToo) {
  basalt::VioConfidenceInputs in;
  in.tracked_ratio = 0.99;
  in.imu_vision_disagreement = true;
  in.numerically_degraded = true;

  auto out = basalt::computeVioConfidence(in);

  EXPECT_DOUBLE_EQ(out.score, 0.0);
  EXPECT_EQ(out.primary_reason, "numerically_degraded");
}

TEST(VioHealthTestSuite, GyroNormIsCarriedThroughButDoesNotAffectScoreYet) {
  // Documented as informational-only until Phase 5's scenario data
  // exists to calibrate a real threshold -- a high rotation rate alone
  // must not currently change the score.
  basalt::VioConfidenceInputs in;
  in.tracked_ratio = 0.95;
  in.triangulated_points = 40;
  in.gyro_norm = 12.0;  // deliberately extreme

  auto out = basalt::computeVioConfidence(in);

  EXPECT_DOUBLE_EQ(out.score, 1.0);
  EXPECT_EQ(out.primary_reason, "nominal");
}
