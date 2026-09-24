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
//
// Loosened 10x from the upstream 1e-3 default: at this rig's focal length
// (fx~455px @ 640x480), 1e-3 rad allows only ~0.45px of deviation from the
// exact epipolar line -- tighter than a consumer OAK-D Lite's combined
// calibration residual + corner-localization jitter + L/R sync skew can
// satisfy, even before descriptor matching runs. Measured triangulated
// yield had been stuck at 5-20% of detected corners all session despite
// the already-loosened Hamming/ratio thresholds above. Confirmed live
// (run 20260921_112133): loosening to 1e-2 (~4.5px tolerance) raised
// average triangulated points/keyframe from 24 to 72 (5.7% -> 15.6% of
// detected corners) on the same rig/environment, validating the
// tolerance-mismatch diagnosis. Kept as the new baseline since.
constexpr double kStereoEpipolarErrorThreshold = 1e-2;

// Keyframe quality gate: minimum triangulated 3D points a keyframe needs
// before it's allowed to become a future match TARGET (see processKeyframe()
// below). Basalt's own keyframe-selection trigger (sqrt_keypoint_vio.cpp,
// vio_new_kf_keypoints_thresh) is a pure motion/parallax heuristic -- it has
// no concept of image quality, so keyframes arrive with wildly inconsistent
// triangulation yield (observed anywhere from under 1% to 20%+ of detected
// corners across a real session).
//
// This must never go below the active config's mapper_min_matches (default
// 20 in vio_config.cpp; the OAK-D live config overrides it to 13): a partner
// with fewer triangulated points than that can *never* produce enough
// PnP-ready matches to pass, no matter how good the descriptor matching is
// (pnp_ready_points is upper-bounded by the partner's own triangulated
// count), so anything below that floor is a pure waste, not a real chance.
//
// History: 30 (initial) -> 15 (too strict, excluded almost everything on
// Pi5) -> 25 -> 20 (current). 15 let through partners whose triangulated
// points were sparse/marginal -- still numerically able to clear
// mapper_min_matches, but a thin or poorly-distributed point set can give
// RANSAC/PnP a badly-conditioned problem, producing a pose that passes
// verification while still being meaningfully wrong. A *wrong* accepted
// closure is worse than a missed one -- it actively pulls the trajectory
// off, rather than just failing to correct it. 25 was a deliberate partial
// revert to isolate whether this specific gate was the dominant cause of
// the increased drift observed after the previous set of loosenings.
// Real low-texture/distant-background sessions (2026-09-17) then hit the
// opposite failure: triangulated points sitting consistently in the
// 10-20 range (stereo match rate ~4.5% vs. ~8.2% on a good background,
// per [STEREO-DIAG] logs) never cleared 25 even once across a whole run,
// so the candidate database stayed empty and num_loop_closures was 0 the
// entire time -- raw drift went completely uncorrected. 20 is a single,
// deliberately modest step down from 25 (NOT a reversion to the
// already-proven-bad 15) to let through the specific 17-20-point
// keyframes observed in that failure case while staying well clear of
// the mapper_min_matches=13 hard floor -- re-check num_loop_closures AND
// whether corrected ends up worse than raw (the signature of accepted-
// but-wrong closures, same failure 15 hit) before loosening further. If
// Pi5 hardware instability keeps triangulation near-zero regardless,
// this alone won't fix it -- that's a data-quality problem no database
// threshold can compensate for.
constexpr int kMinTriangulatedPointsForDatabase = 20;

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

// Minimum time gap (nanoseconds) between a query keyframe and any candidate
// it's allowed to match against. Root-caused via an EuRoC dataset test: with
// no minimum separation, 26 of 29 keyframes in a 15s window "loop closed" --
// nearly every one -- with query/partner index gaps as small as 1 (i.e.
// matching against the immediately preceding keyframe). That isn't a real
// revisit, it's redundantly re-detecting what the odometry chain already
// encodes, and these spurious matches (often high-weight, since near-
// duplicate images produce lots of inliers) measurably made the corrected
// trajectory WORSE than raw against ground truth (ATE RMSE 0.0094m raw vs
// 0.125m corrected on the same run). 2 seconds is long enough that a real
// revisit requires actually having left and come back, not just normal
// keyframe-to-keyframe density.
constexpr int64_t kMinLoopClosureTimeGapNs = 2'000'000'000;

// How many independently-verified closures a single keyframe may
// contribute as pose-graph edges, instead of stopping at the first (as
// before). A node backed by only 1-2 edges has no competing evidence to
// resist a single bad one dragging it off -- the one documented failure
// case (see this file's header, KNOWN LIMITATION) had exactly this
// shape: 2 odometry + 1 bad loop edge, 3 total, nothing else in the
// graph to outvote the bad one with. Multiple independent edges per
// keyframe give the solver real redundancy to outvote an occasional
// wrong closure with.
//
// Was 3; lowered to 2 after a real live Pi5 test froze the whole
// process (--show-gui true) and EuRoC timing confirmed why: even after
// skipping the diagnostic-only countMatchStages() call and the
// expensive optimize_nonlinear() refinement for redundant candidates
// (both real fixes, kept), the still-necessary matchDescriptors() +
// RANSAC cost per extra candidate is not free, and worst-case
// per-keyframe latency was still 1.6-1.8s at 3 -- multiple seconds in
// exactly the "revisiting a well-mapped place" scenario this feature is
// meant to help with. 2 keeps most of the redundancy benefit (a node
// still gets a second, independent piece of evidence instead of just
// one) while meaningfully cutting the worst-case multiplier. Still
// bounded by kMaxCandidatesToVerify regardless of this cap.
constexpr size_t kMaxLoopEdgesPerKeyframe = 2;

// Huber threshold (meters) for robust down-weighting of loop-closure edges
// in solvePoseGraph() -- see the comment at its use site. A residual under
// this is treated as normal noise (full weight); beyond it, weight falls
// off as kLoopEdgeHuberDeltaM / residual_norm, standard IRLS Huber
// behavior. 0.5m is a placeholder starting point (well above normal
// matching/odometry noise, well below the multi-meter residuals a wrong
// long-range match produces), not yet empirically tuned.
constexpr double kLoopEdgeHuberDeltaM = 0.5;

// Drift gate (see OnlineLoopClosure::checkDriftGate()): how far the
// newest keyframe's just-solved corrected position is allowed to diverge
// from what smoothly chaining raw odometry off a FIXED, periodically-
// refreshed anchor node would predict, before the live pose freezes
// instead of advancing onto it. This is deliberately NOT an absolute
// jump-distance threshold (that would false-trigger during genuinely
// fast real motion) -- it's measured against what raw motion alone
// already explains, so it stays near-zero under normal conditions
// regardless of how fast the platform is actually moving, and only
// spikes when a correction is fighting hard against what the raw
// odometry chain says actually happened. Lowered from 1.0m to 0.5m
// after a real live-test session (chronically poor keypoint tracking,
// tracked_ratio rarely above ~0.15) showed forced releases -- see
// kDriftGateMaxHoldSeconds -- snapping over a meter at once: the wider
// threshold was letting more drift accumulate before tripping at all.
// 0.5m is still well above normal solve-to-solve noise (observed
// sub-cm to a few cm on real Pi5 sessions) and below a genuine
// bad-closure jump (multi-meter, per the documented EuRoC case), just
// with less margin than before -- watch for false trips during fast
// real motion if this proves too tight on a better-tracked session.
//
// IMPORTANT: the anchor must be periodically refreshed (see
// kDriftGateAnchorRefreshKeyframes below), not just "the previous node"
// or "N keyframes back" recomputed fresh every check. A real live Pi5
// test caught this: a sliding reference that always recomputes N-back
// only ever measures the CHANGE in (corrected - raw) offset over the
// last N keyframes, never the TOTAL accumulated offset since a fixed
// point -- so a slow, steady creep smaller than threshold/N per keyframe
// can never trip it, no matter how many times it's checked. Confirmed
// on real data: corrected vs raw diverged to over 4m across a 67-second,
// 152-sample episode with no single trip, because the accumulation was
// spread out rather than concentrated. A FIXED anchor, refreshed only
// occasionally during confirmed-good stretches, bounds the worst-case
// undetected drift to whatever accumulates within one refresh interval,
// regardless of whether that happened in one jump or spread across it.
constexpr double kDriftGateThresholdM = 0.5;

// How many consecutive solves must show the residual back under
// kDriftGateThresholdM before the hold releases -- requires the recovery
// to look real, not a single lucky solve amid ongoing instability.
constexpr int kDriftGateReleaseCount = 3;

// How many keyframes the anchor stays fixed before it's allowed to
// refresh forward (only refreshes if the check at that moment still
// passes) -- see kDriftGateThresholdM's comment. This directly bounds
// the worst-case undetected cumulative drift: a slow creep has at most
// this many keyframes to stay under kDriftGateThresholdM before the
// still-fixed anchor catches up with it.
constexpr size_t kDriftGateAnchorRefreshKeyframes = 20;

// Maximum time the drift gate is allowed to hold the live pose before
// force-releasing regardless of whether the residual check ever passes.
// A real live test found it stuck, never confirmed recovered, for 40+
// seconds straight in a genuinely hard case -- for a live display,
// resuming with a possibly-still-imperfect correction is better than
// looking permanently hung. Force-release also refreshes the detection
// anchor to right now (same as a normal release), so detection resumes
// cleanly from this point rather than immediately re-tripping against
// the same stale reference.
//
// Lowered from 30.0 to 15.0, then to 5.0 (2026-09-23): a real Pi5
// rectangle-walk test found checkDriftGate()'s residual check itself is
// structurally slow to resolve during ordinary walking -- see this
// session's investigation, checkDriftGate()'s own header comment. It
// compares the graph's corrected position against a prediction built
// from RAW VIO's dead-reckoning since the hold began; raw VIO drifting
// over real covered distance (the same imprecision loop closure exists
// to correct) means that prediction keeps getting worse the longer a
// hold persists through continued real motion, so a hold tripped mid-leg
// tends to stay tripped for the rest of that leg. Confirmed on that
// test: 4 separate holds each ran the full 15s cap with tracking healthy
// throughout (mean tracked_count 51-74, only 4-11% starved samples) --
// not a tracking failure, the residual genuinely wasn't resolving in
// time. 5.0s matches the existing precedent this codebase already uses
// for "give up and stop suppressing/trust normally" caps elsewhere
// (kBiasFreezeMaxDurationS, kImuVisionReweightMaxDurationS). Extrapolating
// the ~4.8cm/s drift rate this constant's own history above was
// calibrated against, 5s caps the same failure mode to roughly 24cm
// instead of ~70cm. Tradeoff, same shape as the 30->15 change: gives up
// on waiting for a confirmed (kDriftGateReleaseCount consecutive good
// solves) recovery sooner, accepting an unconfirmed correction more
// readily -- untested whether 5s is long enough for genuinely hard cases
// to ever confirm-release at all, watch for an increase in the forced-
// vs-confirmed ratio on a live retest.
constexpr double kDriftGateMaxHoldSeconds = 5.0;

// checkDriftGate() previously only ran when solvePoseGraph() processed a
// NEW accepted closure (need_resolve in processKeyframe()) -- but new
// keyframes keep getting added (with their own raw-derived odometry edge
// to the previous node) regardless of whether a closure was found for
// them, sitting with an un-optimized, still-effectively-raw t_opt until
// the next solve touches them. A live rectangle-walk test found holds
// sitting frozen for their full duration with tracking already healthy
// (mean tracked_count 122.6, 100% nominal for one whole hold window)
// simply because no NEW closure happened to land during that window to
// give checkDriftGate() a chance to re-evaluate. This constant lets
// processKeyframe() force a resolve anyway, on a timer, but ONLY while
// actually held -- there's no equivalent urgency to re-solve when
// nothing's being suppressed, and doing it unconditionally would burn
// real-time budget (pose_graph_solve alone already averages ~64-213ms on
// the Pi5, see OnlineLoopClosure::stop()'s comment) for no live benefit.
// 1.0s gives a hold up to ~5 extra chances to self-resolve within
// kDriftGateMaxHoldSeconds instead of just the one it might otherwise
// get (or zero, if no closure lands at all before the cap).
constexpr double kDriftGatePeriodicRecheckWhileHeldS = 1.0;

// While held, checkDriftGate()'s residual compares the graph's corrected
// position against a prediction built by chaining RAW VIO's own
// dead-reckoning since the hold began (see kDriftGateMaxHoldSeconds) --
// so the reference itself accumulates real error the longer the hold
// persists through continued motion, at roughly the ~4.8cm/s rate that
// constant's own derivation above was calibrated against. A FIXED
// residual bar effectively gets stricter over time purely because of
// that reference's own known unreliability, not because the correction
// is actually any less trustworthy. Growing the held-path threshold by
// this rate compensates for the reference's own expected drift instead
// of penalizing confirmation for it. Deliberately NOT applied to the
// not-held trip-detection comparison (kDriftGateThresholdM there stays
// fixed at 0.5m) -- this is only meant to make genuine recovery easier
// to confirm once already held, not to make tripping in the first place
// any less sensitive. Bounded in practice by kDriftGateMaxHoldSeconds
// (5.0s * 0.048m/s = 0.24m max extra slack, still small next to a
// genuine bad-closure residual, which the documented EuRoC case put at
// multiple meters) rather than an explicit cap constant.
constexpr double kDriftGateHeldThresholdGrowthMPerS = 0.048;

// Starvation trigger: a SECOND way into the same held state above,
// independent of checkDriftGate()'s residual check. That check only
// fires when solvePoseGraph() processes a new keyframe -- but when
// tracking is starved badly enough (something covering the camera, or
// the chronic-low-tracked_ratio failure mode this session kept hitting:
// tracked_ratio down at 2.8-6.5% for tens of seconds on a real live
// test, 2026-09-21), few or no NEW keyframes get created at all, so
// checkDriftGate() may never run during exactly the window it would
// need to. Meanwhile the estimator doesn't "decide" to trust the IMU
// more in that case -- it just has nothing else to lean on, so it
// silently free-integrates while still reporting a confident-looking
// pose. This trigger watches the same tracked_ratio/total_observed_count
// oak_d_vio.cpp already reads for VioConfidenceInputs (see
// reportTrackingHealth()) and enters the SAME held_pose_ state
// checkDriftGate() uses, so it inherits the same max-hold timeout and
// release-blend behavior for free -- this is a second way to TRIP, not
// a parallel hold mechanism.
//
// Deliberately much stricter than VioConfidenceConfig::min_tracked_count
// (8, used for the general "degraded" confidence label, and -- after
// 2026-09-22 -- the SAME value: VioConfidenceConfig's own threshold was
// ratio-based until real data showed it saturating the confidence signal
// near its floor for an entire live run, and was switched to reuse this
// starvation trigger's already-validated absolute-count threshold rather
// than re-deriving a new one). Even though the two now share a value,
// they're checked for different purposes: holding the live pose (this
// trigger) has its own cost (stale data), so it's reserved for genuinely
// critical starvation, while the confidence label is meant to fire at
// the same point tracking quality first becomes suspect. total_observed_
// count is checked separately from tracked_count because if raw
// detections collapse to near-zero (camera fully covered), tracked_count
// alone may not even be a meaningful signal.
// Originally ratio-based (tracked_ratio < 0.10) -- switched to an
// absolute tracked_count floor after live testing (run 20260921_135722)
// showed increasing corner detection density
// (optical_flow_detection_grid_size, num_points_cell -- see
// vio_config.cpp) inflates total_observed_count (the ratio's
// denominator) much faster than it inflates connected/tracked count
// (the numerator): tracked_count improved in absolute terms (avg 16.5
// -> 25.7) but the ratio got WORSE (diluted below 0.10 on 81% of
// frames) because total_observed_count grew ~3.4x. A ratio implicitly
// calibrated against one candidate-pool size breaks any time that size
// changes; absolute tracked_count is what the optimizer actually has to
// work with regardless of how many raw candidates were thrown into the
// detection pool. Still an unvalidated starting guess (picked from this
// run's own bucketed data: ~2-4 during clearly bad stretches, 20+
// during clearly healthy ones) -- needs the same kind of live
// confirmation the ratio threshold it replaces never got either.
constexpr int kStarvationMinTrackedCount = 8;
constexpr int kStarvationMinTotalObserved = 5;
// Wall-clock, not a call count, matching drift_held_since_wall_'s own
// reasoning: getSmoothedCorrectedPose() is called from several sites in
// oak_d_vio.cpp (GUI draw, dashboard publish, etc.), not at one single
// guaranteed rate, so counting consecutive CALLS would make this
// trigger's sensitivity depend on how many consumers happen to be
// active, not on how long the starvation actually lasted.
constexpr double kStarvationPersistenceSeconds = 2.0;
// How long a RECOVERY (tracked_count/total_observed_count back above
// threshold) must itself persist before the starvation clock actually
// resets -- see has_good_streak_'s comment in the .h. Deliberately
// shorter than kStarvationPersistenceSeconds (confirming a real recovery
// should be faster/easier than confirming a real starvation episode),
// but long enough to reject the sub-second flickers a real live test
// (2026-09-22, untextured wall) showed recurring every 0.3-0.7s through
// an extended bad stretch -- picked as roughly double the longest single
// flicker observed there (~0.4s), not independently tuned.
constexpr double kStarvationRecoveryGraceS = 1.0;

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

// One verified match accepted during a single keyframe's candidate loop
// -- see kMaxLoopEdgesPerKeyframe above for why more than one can be
// accepted per keyframe now, instead of stopping at the first.
struct AcceptedClosure {
  size_t partner_idx;
  Sophus::SE3d T_body_partner_new;
  int num_inliers;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace

OnlineLoopClosure::OnlineLoopClosure(const Calibration<double>& calib,
                                     const VioConfig& config)
    : calib_(calib), config_(config) {
  hash_bow_.reset(new HashBow<256>(config_.mapper_bow_num_bits));
  input_queue.set_capacity(1000);
  localization_queue.set_capacity(1000);
  drift_gate_events.set_capacity(1000);
}

OnlineLoopClosure::~OnlineLoopClosure() { stop(); }

void OnlineLoopClosure::start() {
  if (running.exchange(true)) return;
  worker_thread_ = std::thread(&OnlineLoopClosure::processingLoop, this);
}

void OnlineLoopClosure::stop() {
  if (!running.exchange(false)) return;

  // Discard whatever's still queued instead of draining it. On a live
  // Pi5 run, loop_candidate_search+pose_graph_solve averaged ~289ms/kf
  // against a ~285ms average keyframe interval -- essentially 100% of
  // real-time budget with no slack, so any jitter puts this thread
  // permanently behind for the rest of the run. Pushing nullptr to the
  // back of input_queue (the old behavior) meant Ctrl+C had to fully
  // process that entire backlog before the app could exit -- confirmed
  // live: 81 keyframes still queued after the rest of the app had
  // already shut down, ~30s of candidate-search+pose-graph-solve alone.
  // By the time stop() runs here, the producer (vio->maybe_join(),
  // called before this in oak_d_vio.cpp) has already stopped pushing,
  // so draining is safe -- nothing new can arrive after it. Whatever's
  // still queued at shutdown was never going to be scored in real time
  // anyway, so dropping it costs nothing the run didn't already accept;
  // it only bounds shutdown to whatever single keyframe is already
  // in-flight inside processKeyframe() (worst case ~1s), not the whole
  // backlog.
  MargData::Ptr discard;
  while (input_queue.try_pop(discard)) {
  }
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

  // Fold in the VIO estimator's own multi-view-triangulated landmarks
  // (see MargData::host_landmark_px/host_landmark_pt3d's comment) as
  // EXTRA corners, before angle/descriptor computation below so they
  // get descriptors through the exact same code path as independently-
  // detected ones. Dedup against corners detectKeypointsMapping already
  // found (a VIO-tracked point and a freshly-detected one can easily
  // land on the same physical corner) and skip anything too close to
  // the border for computeAngles()/computeDescriptors()'s patch access
  // to stay in-bounds -- same EDGE_THRESHOLD margin
  // detectKeypointsMapping's own corners are filtered to (see
  // src/utils/keypoints.cpp).
  //
  // Live-validated 2026-09-24 (see kMaxHarvestedLandmarksPerKf's comment
  // in sqrt_keypoint_vio.cpp for the full A/B writeup): this is what
  // lets a genuinely-matched long-range closure back to the true start
  // actually clear mapper_min_matches, instead of being capped by a
  // candidate keyframe's own sparse single-instant stereo triangulation.
  // (corner index in kf.kd0.corners, VIO-landmark 3D point in this
  // keyframe's own cam0 frame) for every landmark corner actually kept
  // below -- applied to kf.pts3d/corner_to_pt3d AFTER the stereo-
  // triangulation block further down, so a VIO-sourced 3D point (much
  // stronger baseline -- see this block's header comment) takes
  // precedence over anything the single-instant stereo match also found
  // for the same corner, rather than the other way around.
  std::vector<std::pair<int, Eigen::Vector3d>> vio_landmark_pts3d;
  {
    constexpr double kDedupRadiusPx = 3.0;
    constexpr int kEdgeThresholdPx = 19;
    for (size_t li = 0; li < data->host_landmark_px.size(); li++) {
      const Eigen::Vector2d& px = data->host_landmark_px[li];
      if (!img0.InBounds((float)px.x(), (float)px.y(), kEdgeThresholdPx))
        continue;
      // Reuse an existing nearby corner's index instead of discarding the
      // landmark -- a live test found ~97% of harvested landmarks land
      // within a few px of a corner detectKeypointsMapping() also found
      // (both detectors picking out the same salient points), so
      // dropping on any overlap wasted almost the entire feature (2.7%
      // utilization: 1008/37884 landmarks used across one run). Most of
      // those coinciding corners never got a 3D point from the ~10%-yield
      // stereo match anyway (see kStereoEpipolarErrorThreshold's
      // comment), so attaching the landmark's point to the SAME corner
      // index -- rather than only to a brand-new one -- is what actually
      // captures the opportunity.
      int idx = -1;
      for (size_t ci = 0; ci < kf.kd0.corners.size(); ci++) {
        if ((kf.kd0.corners[ci] - px).squaredNorm() <
            kDedupRadiusPx * kDedupRadiusPx) {
          idx = (int)ci;
          break;
        }
      }
      if (idx < 0) {
        idx = (int)kf.kd0.corners.size();
        kf.kd0.corners.push_back(px);
      }
      vio_landmark_pts3d.emplace_back(idx, data->host_landmark_pt3d[li]);
    }
  }

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

    // Apply the VIO-landmark 3D points collected earlier (see this
    // function's kDedupRadiusPx block) -- done here, after stereo
    // triangulation above, so a corner that BOTH the single-instant
    // stereo match and a VIO landmark cover ends up pointing at the
    // VIO-sourced entry (stronger multi-view baseline), not whichever
    // ran first. Always appends a fresh kf.pts3d entry rather than
    // trying to overwrite one in place -- corner_to_pt3d is a map, so
    // repointing it here just leaves the stereo-sourced entry (if any)
    // unreferenced, which is harmless.
    int vio_landmark_pts3d_used = 0;
    for (const auto& [corner_idx, pt3d] : vio_landmark_pts3d) {
      int pt_idx = (int)kf.pts3d.size();
      kf.pts3d.push_back(pt3d);
      kf.corner_to_pt3d[corner_idx] = pt_idx;
      vio_landmark_pts3d_used++;
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
              << " triangulated=" << kf.pts3d.size()
              << " (vio_landmarks=" << vio_landmark_pts3d_used << "/"
              << data->host_landmark_px.size() << ")" << std::endl;

    // Exposed for a live confidence signal (vio_health.h) / scenario-
    // characterization tooling -- this line was already printed every
    // keyframe, just never retained anywhere queryable.
    latest_triangulated_points = (int)kf.pts3d.size();
  }

  auto t2 = std::chrono::steady_clock::now();

  // --- Loop detection: query BoW DB restricted to earlier keyframes ---
  // Up to kMaxLoopEdgesPerKeyframe independently-verified matches get
  // accepted here, not just the first -- see that constant's comment.
  Eigen::aligned_vector<AcceptedClosure> accepted_closures;

  {
    std::vector<std::pair<TimeCamId, double>> results;
    hash_bow_->querry_database(kf.kd0.bow_vector,
                               (size_t)config_.mapper_num_frames_to_match,
                               results, &kf_id);

    std::vector<std::pair<TimeCamId, double>> above_threshold;
    for (const auto& cand : results) {
      if (cand.second >= config_.mapper_frames_to_match_threshold &&
          kf_id - cand.first.frame_id >= kMinLoopClosureTimeGapNs) {
        above_threshold.push_back(cand);
      }
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
      if (accepted_closures.size() >= kMaxLoopEdgesPerKeyframe) break;

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

      // Whether this is the first candidate this keyframe attempts --
      // reused below to also skip the (real, not diagnostic-only) non-
      // linear refinement for redundant candidates. Computed once here
      // since it doesn't change within this iteration.
      bool is_first_candidate_attempt = accepted_closures.empty();

      // countMatchStages() is diagnostic-only logging (see its own
      // comment -- doesn't affect the actual matches used downstream),
      // but it's a full O(corners0 x corners_partner) brute-force pass,
      // same cost class as matchDescriptors() right below. Multi-edge
      // redundancy means up to kMaxCandidatesToVerify candidates can
      // reach this point per keyframe regardless of how many end up
      // accepted/refined, so running this a second time per candidate
      // for pure logging was real, unnecessary cost contributing to the
      // multi-second per-keyframe stalls found on a live Pi5 test (see
      // the refinement-skip fix above this loop). Only pay for it on the
      // first candidate, where the detailed breakdown is most useful
      // anyway (it's the one most likely to actually get accepted).
      int after_hamming = -1, after_ratio = -1;
      if (is_first_candidate_attempt) {
        countMatchStages(kf.kd0.corner_descriptors, partner->kd0.corner_descriptors,
                         (int)config_.mapper_max_hamming_distance,
                         config_.mapper_second_best_test_ratio, after_hamming,
                         after_ratio);
      }

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
                << "              matches_after_hamming="
                << (after_hamming < 0 ? "n/a (skipped, redundant candidate)"
                                      : std::to_string(after_hamming))
                << "\n"
                << "              matches_after_ratio_test="
                << (after_ratio < 0 ? "n/a" : std::to_string(after_ratio))
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
      // avoids that cost on every failed attempt. Refinement itself is
      // additionally capped to at most once per keyframe regardless of
      // how many candidates pass this check -- see
      // is_first_candidate_attempt below.
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
      //
      // Only for the FIRST accepted candidate this keyframe (candidates
      // are processed in BoW-score order, so that's the best-scoring
      // one) -- multi-edge redundancy (kMaxLoopEdgesPerKeyframe) means up
      // to kMaxLoopEdgesPerKeyframe candidates can reach this point per
      // keyframe, and optimize_nonlinear() doesn't scale with that: a
      // real live test measured up to 3.1s for a SINGLE keyframe's
      // candidate loop once multiple candidates each paid for their own
      // refinement (73-455ms each, per the pre-refinement-reject comment
      // above) -- exactly the "revisiting a well-mapped place" scenario
      // multi-edge is meant to help with, made unusably slow by it
      // instead, badly stalling a live Pi5 session. Later (redundant)
      // candidates use the raw RANSAC model directly -- it already
      // cleared the same inlier-count floor just above. Redundancy's
      // whole point is resisting a bad edge with OTHER evidence, not
      // needing every piece of that evidence to be maximally precise.
      if (is_first_candidate_attempt) {
        adapter.sett(ransac.model_coefficients_.topRightCorner<3, 1>());
        adapter.setR(ransac.model_coefficients_.topLeftCorner<3, 3>());
        opengv::transformation_t refined =
            opengv::absolute_pose::optimize_nonlinear(adapter, ransac.inliers_);
        ransac.sac_model_->selectWithinDistance(refined, ransac.threshold_,
                                                ransac.inliers_);
        ransac.model_coefficients_ = refined;

        if ((int)ransac.inliers_.size() < config_.mapper_min_matches) {
          std::cout << "              RESULT: rejected (refined inliers "
                    << ransac.inliers_.size() << " < mapper_min_matches "
                    << config_.mapper_min_matches << ")" << std::endl;
          continue;
        }
      }

      Eigen::Vector3d ransac_t =
          ransac.model_coefficients_.topRightCorner<3, 1>();
      std::cout << "              ransac_inliers=" << ransac.inliers_.size()
                << (is_first_candidate_attempt ? " (refined)" : " (raw, redundant edge)")
                << "  ransac_pose_t=[" << ransac_t.x() << ", "
                << ransac_t.y() << ", " << ransac_t.z() << "]" << std::endl;

      Sophus::SE3d T_partnerCam_newCam(
          ransac.model_coefficients_.topLeftCorner<3, 3>(),
          ransac.model_coefficients_.topRightCorner<3, 1>());

      AcceptedClosure closure;
      closure.partner_idx = partner_idx;
      closure.T_body_partner_new =
          calib_.T_i_c[0] * T_partnerCam_newCam * calib_.T_i_c[0].inverse();
      closure.num_inliers = (int)ransac.inliers_.size();
      accepted_closures.push_back(closure);

      // Publish the localization result immediately -- before the
      // pose-graph insertion/solve below -- so consumers that need "where
      // am I now" (navigation, RTL) don't have to wait on the potentially
      // slower global solve. Composed against the reference keyframe's
      // current CORRECTED pose (already reflecting any earlier loop
      // closures), not its raw VIO pose, so T_w_current is already
      // globally-consistent without needing a fresh solve first. Published
      // for every accepted closure, not just one -- a consumer wanting
      // "the freshest single answer" can just take the latest off the
      // queue, same as before.
      {
        LocalizationResult loc;
        loc.t_ns = kf_id;
        loc.reference_t_ns = partner_t_ns;
        loc.T_reference_current = closure.T_body_partner_new;
        Sophus::SE3d T_w_reference_corrected(
            composeYPR(partner->roll, partner->pitch, partner->yaw),
            partner->t_opt);
        loc.T_w_current = T_w_reference_corrected * closure.T_body_partner_new;
        loc.num_inliers = closure.num_inliers;
        localization_queue.try_push(loc);
      }

      std::cout << "              RESULT: ACCEPTED (edge " << accepted_closures.size()
                << " of up to " << kMaxLoopEdgesPerKeyframe
                << " for this keyframe)" << std::endl;
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

    for (const auto& closure : accepted_closures) {
      PoseGraphEdge e;
      e.i = closure.partner_idx;
      e.j = new_idx;
      e.dt = closure.T_body_partner_new.translation();
      double r, p, y;
      decomposeYPR(closure.T_body_partner_new.rotationMatrix(), r, p, y);
      e.dyaw = y;
      // See PoseGraphEdge::weight comment (online_loop_closure.h) -- a
      // closure that just barely cleared mapper_min_matches gets the same
      // baseline trust as an odometry edge (weight 1.0); one with several
      // times as many inliers pulls proportionally harder, capped at 5x so
      // a single very-strong match can't dominate the graph unboundedly.
      e.weight = std::clamp(
          closure.num_inliers / std::max(1.0, config_.mapper_min_matches), 1.0,
          5.0);
      e.is_loop = true;
      edges_.push_back(e);

      num_loop_closures++;
      need_resolve = true;
    }

    if (!accepted_closures.empty()) {
      std::cout << "              (graph updated, " << accepted_closures.size()
                << " loop edge(s) added this keyframe, total_closures="
                << num_loop_closures.load() << ")" << std::endl;
    }

    t_ns_to_idx_[kf_id] = new_idx;
    keyframes_.push_back(std::move(kf));

    // Periodic while-held recheck -- see kDriftGatePeriodicRecheckWhileHeldS.
    // Gives checkDriftGate() a chance to re-evaluate using whatever's
    // already in the graph (this new keyframe's own odometry edge plus
    // any earlier loop-closure edges) even when no NEW closure was found
    // for THIS keyframe specifically.
    if (!need_resolve && drift_held_) {
      double since_last_resolve = std::chrono::duration<double>(
                                       std::chrono::steady_clock::now() -
                                       last_resolve_wall_)
                                       .count();
      if (since_last_resolve >= kDriftGatePeriodicRecheckWhileHeldS) {
        need_resolve = true;
      }
    }
  }

  auto t4 = std::chrono::steady_clock::now();

  // Quality gate (see kMinTriangulatedPointsForDatabase above): this
  // keyframe still got a pose-graph node and a chance to QUERY against
  // history above (we don't get to choose which frames Basalt hands us),
  // but only keyframes with enough triangulated points are added as future
  // match TARGETS, so a poor one can't poison later keyframes' candidate
  // pool the way we saw happen repeatedly during live testing.
  //
  // The very first keyframe this instance ever processes is stored
  // unconditionally as a dedicated "home" reference, bypassing the quality
  // gate -- requiring it to also clear the normal bar risks having NO home
  // reference at all if triangulation hasn't stabilized yet right at
  // startup (camera still settling), which would defeat the purpose.
  size_t num_pts3d = keyframes_.back().pts3d.size();
  bool quality_ok = (int)num_pts3d >= kMinTriangulatedPointsForDatabase;
  bool is_home_keyframe = keyframes_.size() == 1;

  if (quality_ok || is_home_keyframe) {
    hash_bow_->add_to_database(TimeCamId(kf_id, 0),
                               keyframes_.back().kd0.bow_vector);
    if (is_home_keyframe) {
      home_keyframe_t_ns_ = kf_id;
      std::cout << "[ONLINE-LOOP] kf=0 (t_ns=" << kf_id
                << ") captured as HOME reference keyframe (" << num_pts3d
                << " triangulated points"
                << (quality_ok ? "" : ", below quality gate but stored anyway")
                << ")" << std::endl;
    }
  } else {
    std::cout << "[ONLINE-LOOP] kf=" << (keyframes_.size() - 1)
              << " excluded from candidate database (" << num_pts3d
              << " triangulated points < " << kMinTriangulatedPointsForDatabase
              << ")" << std::endl;
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

  // See kDriftGatePeriodicRecheckWhileHeldS -- marks "a resolve just
  // happened" regardless of why this call was triggered (new closure or
  // periodic while-held recheck), so the next periodic recheck is timed
  // from here, not from the last NEW-closure-triggered solve.
  last_resolve_wall_ = std::chrono::steady_clock::now();

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

      // Robust (Huber) down-weighting for LOOP edges only, recomputed each
      // iteration from the CURRENT residual (standard IRLS pattern). An
      // edge whose translation residual disagrees with the rest of the
      // graph by more than kLoopEdgeHuberDeltaM gets progressively less
      // say the worse it disagrees, instead of being trusted at full
      // weight or rejected outright before ever being tried. Root-caused
      // via EuRoC testing: pre-filtering candidates by inlier count or
      // time gap either let wrong long-range matches through (aliasing in
      // a visually repetitive room) or blocked genuine ones alongside them
      // (a true distant revisit naturally has fewer inliers too, since
      // more time means more viewpoint change even when correct) -- inlier
      // count can't reliably separate the two cases in advance. Odometry
      // edges are deliberately NOT robustified: they're the trusted
      // backbone that should resist being dragged by a bad loop edge, not
      // get weakened right alongside it.
      double robust_scale = 1.0;
      if (e.is_loop) {
        double res_norm = r.head<3>().norm();
        if (res_norm > kLoopEdgeHuberDeltaM) {
          robust_scale = kLoopEdgeHuberDeltaM / res_norm;
        }
      }
      double w = e.weight * robust_scale;

      // Scaling the per-edge contribution by w before accumulation is
      // equivalent to minimizing sum_e w_e * ||r_e||^2 -- the normal
      // equations become sum_e w_e * J_e^T J_e * dx = -sum_e w_e * J_e^T
      // r_e, same Gauss-Newton derivation as before, just weighted.
      Eigen::Matrix4d Jii = w * J.block<4, 4>(0, 0).transpose() * J.block<4, 4>(0, 0);
      Eigen::Matrix4d Jjj = w * J.block<4, 4>(0, 4).transpose() * J.block<4, 4>(0, 4);
      Eigen::Matrix4d Jij = w * J.block<4, 4>(0, 0).transpose() * J.block<4, 4>(0, 4);

      if (oi >= 0) {
        add_block(oi, oi, Jii);
        Jtr.segment(oi, 4) += w * J.block<4, 4>(0, 0).transpose() * r;
      }
      if (oj >= 0) {
        add_block(oj, oj, Jjj);
        Jtr.segment(oj, 4) += w * J.block<4, 4>(0, 4).transpose() * r;
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

  checkDriftGate();
}

// Called only from solvePoseGraph(), which already holds state_mutex_ --
// must not try to re-lock it (non-recursive mutex, would deadlock).
//
// See kDriftGateThresholdM's comment for the full reasoning. Short
// version: compares the newest keyframe's just-solved corrected position
// against what chaining its real raw-VIO motion off a FIXED reference
// (drift_anchor_idx_, refreshed only periodically -- see
// kDriftGateAnchorRefreshKeyframes) would predict. Not held: check
// against drift_anchor_idx_, and refresh it forward every
// kDriftGateAnchorRefreshKeyframes keyframes if that check still passes.
// Held: check against the frozen held_pose_/held_anchor_raw_pose_
// instead -- deliberately asymmetric (quick to trip, more carefully
// verified to release), which is the right shape for a safety gate.
// Note: drift_anchor_idx_ is only ever used to DETECT a trip -- what
// actually gets frozen (held_pose_) comes from last_published_pose_
// instead, not from the detection anchor's own node (see that member's
// header comment for why).
void OnlineLoopClosure::checkDriftGate() {
  size_t n = keyframes_.size();
  if (n < 2) return;
  if (drift_anchor_idx_ >= n) drift_anchor_idx_ = 0;  // safety, shouldn't happen

  const LoopKeyframe& newest = keyframes_[n - 1];
  Sophus::SE3d T_newest_corrected(composeYPR(newest.roll, newest.pitch, newest.yaw),
                                  newest.t_opt);

  // Force-release regardless of residual once the hold has gone on too
  // long -- see kDriftGateMaxHoldSeconds's comment. Checked before the
  // normal residual-based path since a live display shouldn't stay
  // stuck indefinitely waiting for the graph to look right again.
  if (drift_held_ && drift_held_since_t_ns_ >= 0) {
    double held_seconds = (newest.t_ns - drift_held_since_t_ns_) / 1e9;
    if (held_seconds >= kDriftGateMaxHoldSeconds) {
      forceReleaseDriftHoldLocked();
      std::cout << "[ONLINE-LOOP] DRIFT GATE FORCE-RELEASED: kf=" << (n - 1)
                << " after " << held_seconds << "s (max hold "
                << kDriftGateMaxHoldSeconds
                << "s exceeded) -- resuming with unconfirmed correction"
                << std::endl;
      return;
    }
  }

  Sophus::SE3d T_reference_corrected;
  Sophus::SE3d T_reference_raw;
  if (drift_held_) {
    T_reference_corrected = held_pose_;
    T_reference_raw = held_anchor_raw_pose_;
  } else {
    const LoopKeyframe& anchor = keyframes_[drift_anchor_idx_];
    T_reference_corrected =
        Sophus::SE3d(composeYPR(anchor.roll, anchor.pitch, anchor.yaw), anchor.t_opt);
    T_reference_raw = anchor.T_w_i_raw;
  }

  Sophus::SE3d T_raw_delta = T_reference_raw.inverse() * newest.T_w_i_raw;
  Sophus::SE3d T_predicted = T_reference_corrected * T_raw_delta;
  double residual_m =
      (T_newest_corrected.translation() - T_predicted.translation()).norm();

  // See kDriftGateHeldThresholdGrowthMPerS -- only grows the bar on the
  // held (confirm-to-release) path, never the not-held trip-detection
  // one, so tripping stays exactly as sensitive as before.
  double effective_threshold_m = kDriftGateThresholdM;
  if (drift_held_) {
    double held_wall_s = std::chrono::duration<double>(
                              std::chrono::steady_clock::now() -
                              drift_held_since_wall_)
                              .count();
    effective_threshold_m += kDriftGateHeldThresholdGrowthMPerS * held_wall_s;
  }

  if (residual_m > effective_threshold_m) {
    if (!drift_held_) {
      // Freeze at wherever the live pose actually was an instant ago
      // (last_published_pose_/last_raw_pose_seen_, cached by
      // getSmoothedCorrectedPose()), NOT at the detection anchor's node
      // -- see held_pose_'s header comment. The anchor can be far
      // enough behind (even still keyframe 0, the VIO world origin) that
      // freezing there would teleport the live pose backwards instead of
      // holding it in place. Falls back to the anchor only in the
      // practically-unreachable case that no live pose was ever
      // published before the very first trip.
      if (have_last_published_pose_) {
        held_pose_ = last_published_pose_;
        held_anchor_raw_pose_ = last_raw_pose_seen_;
      } else {
        held_pose_ = T_reference_corrected;
        held_anchor_raw_pose_ = T_reference_raw;
      }
      drift_held_ = true;
      drift_held_since_t_ns_ = newest.t_ns;
      drift_held_since_wall_ = std::chrono::steady_clock::now();
      drift_gate_events.try_push(DriftGateEvent::kTripped);
      std::cout << "[ONLINE-LOOP] DRIFT GATE TRIPPED: kf=" << (n - 1)
                << " residual=" << residual_m
                << "m (threshold=" << kDriftGateThresholdM
                << "m, detected vs anchor kf=" << drift_anchor_idx_
                << ") -- holding live pose in place" << std::endl;
    }
    drift_gate_stable_count_ = 0;
    return;
  }

  if (drift_held_) {
    if (++drift_gate_stable_count_ >= kDriftGateReleaseCount) {
      drift_held_ = false;
      drift_gate_stable_count_ = 0;
      // Recovery also refreshes the anchor to right now -- resume normal
      // tracking from the point just confirmed good, not the stale
      // pre-trip anchor.
      drift_anchor_idx_ = n - 1;
      keyframes_since_anchor_refresh_ = 0;
      release_blending_ = true;
      release_blend_start_pose_ = held_pose_;
      release_blend_start_wall_ = std::chrono::steady_clock::now();
      release_blend_duration_s_ = -1.0;  // recomputed on first use -- see comment
      // See forceReleaseDriftHoldLocked()'s matching reset -- same bug,
      // same fix, for the confirmed-release path.
      starvation_active_ = false;
      has_good_streak_ = false;
      drift_gate_events.try_push(DriftGateEvent::kReleasedConfirmed);
      std::cout << "[ONLINE-LOOP] DRIFT GATE RELEASED: kf=" << (n - 1)
                << " residual=" << residual_m
                << "m (effective_threshold=" << effective_threshold_m
                << "m) back under threshold for " << kDriftGateReleaseCount
                << " consecutive solves" << std::endl;
    }
    return;
  }

  // Not held and this check passed -- eligible to refresh the anchor
  // forward, but only after a real stretch of confirmed-good keyframes
  // (see kDriftGateAnchorRefreshKeyframes), not on every passing check.
  // Refreshing every time would recreate exactly the sliding-reference
  // bug this design replaced: a slow, steady creep would keep resetting
  // the accumulator before it ever crossed the threshold.
  if (++keyframes_since_anchor_refresh_ >= kDriftGateAnchorRefreshKeyframes) {
    drift_anchor_idx_ = n - 1;
    keyframes_since_anchor_refresh_ = 0;
  }
}

bool OnlineLoopClosure::isDriftHeld() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return drift_held_;
}

void OnlineLoopClosure::forceReleaseDriftHoldLocked() const {
  drift_held_ = false;
  drift_gate_stable_count_ = 0;
  drift_anchor_idx_ = keyframes_.empty() ? 0 : keyframes_.size() - 1;
  keyframes_since_anchor_refresh_ = 0;
  release_blending_ = true;
  release_blend_start_pose_ = held_pose_;
  release_blend_start_wall_ = std::chrono::steady_clock::now();
  release_blend_duration_s_ = -1.0;  // recomputed on first use -- see comment
  last_forced_release_wall_ = release_blend_start_wall_;
  had_forced_release_ = true;
  // Reset the starvation trigger's clock too (see
  // kStarvationMinTrackedCount in the .cpp): without this, a release
  // followed immediately by still-bad tracking would re-trip on the very
  // next check using the OLD, un-reset elapsed time, chaining what should
  // be separate hold episodes into what looks like one continuous freeze
  // -- confirmed on a real live test (2026-09-21): reported "starved for"
  // durations of 10-23s despite kStarvationPersistenceSeconds being 2.0,
  // because this reset was missing.
  starvation_active_ = false;
  has_good_streak_ = false;
  drift_gate_events.try_push(DriftGateEvent::kReleasedForced);
}

bool OnlineLoopClosure::isRecentlyForceReleased() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (!had_forced_release_) return false;
  double elapsed_s = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() -
                          last_forced_release_wall_)
                          .count();
  return elapsed_s < kForcedReleaseCooldownS;
}

void OnlineLoopClosure::reportTrackingHealth(double tracked_ratio,
                                             int tracked_count,
                                             int total_observed_count) {
  latest_reported_tracked_ratio_ = tracked_ratio;
  latest_reported_tracked_count_ = tracked_count;
  latest_reported_total_observed_count_ = total_observed_count;
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

bool OnlineLoopClosure::getSmoothedCorrectedPose(
    const Sophus::SE3d& current_raw_pose, Sophus::SE3d& out) const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (keyframes_.empty()) return false;

  // Drift gate (see checkDriftGate()): while held, the live pose stays
  // fixed at the last trusted point instead of advancing -- even with
  // real raw motion since then -- rather than following a newly-solved
  // position whose odometry-edge residual looks implausible. The graph
  // keeps solving normally in the background regardless (processKeyframe
  // isn't gated by this), so it still gets a chance to self-correct;
  // only what's published live is held back until it does.
  if (drift_held_) {
    // Wall-clock watchdog for kDriftGateMaxHoldSeconds -- see
    // drift_held_since_wall_'s comment. checkDriftGate()'s own timeout
    // check only runs when solvePoseGraph() processes a new keyframe; if
    // tracking is bad enough that keyframes stop arriving, that check
    // never gets a chance to fire. This one runs here instead, since
    // getSmoothedCorrectedPose() is called at full pose-publish rate
    // regardless of keyframe activity. Falls through to the normal
    // (not-held) path below on release, rather than returning
    // held_pose_ one more time, so this call already produces a fresh
    // (blended) pose instead of waiting for the next one.
    double held_seconds_wall = std::chrono::duration<double>(
                                    std::chrono::steady_clock::now() -
                                    drift_held_since_wall_)
                                    .count();
    if (held_seconds_wall >= kDriftGateMaxHoldSeconds) {
      forceReleaseDriftHoldLocked();
      std::cout << "[ONLINE-LOOP] DRIFT GATE FORCE-RELEASED (wall-clock "
                   "watchdog, no recent keyframe activity): held for "
                << held_seconds_wall << "s" << std::endl;
    } else {
      out = held_pose_;
      return true;
    }
  }

  // Starvation trigger (see kStarvationMinTrackedCount in the .cpp) --
  // a second way into drift_held_, independent of checkDriftGate()'s
  // residual check, for when tracking is starved badly enough (camera
  // covered, or the chronic-low-tracked-count failure mode) that few or
  // no new keyframes are being created at all, so that check may never
  // get a chance to run. At this point drift_held_ is definitely false
  // (either it always was, or the block above just released it), so this
  // only evaluates whether to trip a NEW hold, never conflicts with an
  // in-progress one.
  bool starved =
      latest_reported_tracked_count_ < kStarvationMinTrackedCount ||
      latest_reported_total_observed_count_ < kStarvationMinTotalObserved;
  auto now = std::chrono::steady_clock::now();
  if (starved) {
    // A real recovery is in progress (if any) -- cancel it. This is what
    // makes a brief good flicker unable to discard accumulated bad-streak
    // time: has_good_streak_ only matters once it's persisted past
    // kStarvationRecoveryGraceS in the (!starved) branch below, and a
    // single starved sample here resets it before it gets the chance.
    has_good_streak_ = false;
    if (!starvation_active_) {
      starvation_active_ = true;
      starvation_since_wall_ = now;
    }
    double starved_s =
        std::chrono::duration<double>(now - starvation_since_wall_).count();
    if (starved_s >= kStarvationPersistenceSeconds) {
      // Freeze at wherever the live pose actually was an instant ago,
      // same reasoning as checkDriftGate()'s own trip -- see held_pose_'s
      // header comment for why last_published_pose_, not some graph node.
      if (have_last_published_pose_) {
        held_pose_ = last_published_pose_;
        held_anchor_raw_pose_ = last_raw_pose_seen_;
      } else {
        const LoopKeyframe& newest = keyframes_.back();
        held_pose_ = Sophus::SE3d(composeYPR(newest.roll, newest.pitch, newest.yaw),
                                  newest.t_opt);
        held_anchor_raw_pose_ = newest.T_w_i_raw;
      }
      drift_held_ = true;
      drift_held_since_t_ns_ = -1;  // no keyframe backing this trip -- see
                                     // drift_held_since_wall_'s comment, the
                                     // wall-clock twin is what actually
                                     // governs the max-hold cap here.
      drift_held_since_wall_ = now;
      drift_gate_events.try_push(DriftGateEvent::kTripped);
      std::cout << "[ONLINE-LOOP] DRIFT GATE TRIPPED (starvation: "
                   "tracked_count="
                << latest_reported_tracked_count_
                << " tracked_ratio=" << latest_reported_tracked_ratio_
                << " total_observed_count="
                << latest_reported_total_observed_count_
                << ", starved for " << starved_s
                << "s) -- holding live pose in place" << std::endl;
      out = held_pose_;
      return true;
    }
  } else if (starvation_active_) {
    // Only reset the bad-streak clock once this healthy sample's streak
    // has ITSELF persisted for kStarvationRecoveryGraceS -- see
    // has_good_streak_'s comment in the .h.
    if (!has_good_streak_) {
      has_good_streak_ = true;
      good_streak_since_wall_ = now;
    }
    double good_s =
        std::chrono::duration<double>(now - good_streak_since_wall_).count();
    if (good_s >= kStarvationRecoveryGraceS) {
      starvation_active_ = false;
      has_good_streak_ = false;
    }
  }

  const LoopKeyframe& kf = keyframes_.back();
  Sophus::SE3d T_w_i_corrected_kf(composeYPR(kf.roll, kf.pitch, kf.yaw),
                                   kf.t_opt);
  // Raw motion since this keyframe was captured -- kf.T_w_i_raw and
  // current_raw_pose are both raw VIO poses in the same (uncorrected)
  // world frame, so this delta is meaningful even though that frame's
  // origin/yaw is arbitrary.
  Sophus::SE3d T_kf_to_current = kf.T_w_i_raw.inverse() * current_raw_pose;
  out = T_w_i_corrected_kf * T_kf_to_current;

  // Blend across ordinary target movement too -- see
  // last_unblended_target_pose_'s header comment for why this has to
  // compare against the last UNBLENDED target, not last_published_pose_
  // (which deliberately lags behind target while a blend is in flight,
  // and would look like continuous movement every tick if used here).
  // Arms (or RESTARTS, from wherever the stream actually is right now,
  // i.e. last_published_pose_ -- correct whether that's a steady pose or
  // mid-blend from an earlier jump/release) whenever the fresh target
  // meaningfully disagrees with the previous fresh target. Deliberately
  // does NOT skip when a blend is already in flight: that was tried
  // first and measured live (run 20260921_114840) to let some jumps leak
  // through nearly full-strength (up to ~40cm in one tick) whenever they
  // landed while a prior blend's alpha was already close to 1, since the
  // in-flight blend's target silently absorbed the new jump with no
  // re-arm to smooth it.
  if (have_last_unblended_target_pose_) {
    double jump_m = (out.translation() -
                     last_unblended_target_pose_.translation())
                        .norm();
    if (jump_m > kTargetJumpBlendThresholdM) {
      release_blending_ = true;
      release_blend_start_pose_ =
          have_last_published_pose_ ? last_published_pose_ : out;
      release_blend_start_wall_ = std::chrono::steady_clock::now();
      release_blend_duration_s_ = -1.0;  // recomputed on first use below
    }
  }
  last_unblended_target_pose_ = out;
  have_last_unblended_target_pose_ = true;

  // After a hold releases OR an ordinary anchor switch (either way --
  // see checkDriftGate() and release_blending_'s header comment), glide
  // from where the pose was frozen/last published to the freshly-
  // computed target instead of jumping to it in one step. SE3 geodesic
  // interpolation (linear on translation, exponential-map on the log of
  // the relative rotation) so the blend is a single consistent
  // rigid-body motion, not independently lerped translation/rotation.
  if (release_blending_) {
    // Duration is computed once per blend, on the first tick that
    // actually sees both endpoints (every arm site just sets this to -1
    // and lets this be the single place that derives it) -- rate-limited
    // by kMaxBlendSpeedMps rather than always kReleaseBlendDurationS, so
    // a big correction gets stretched out instead of moving implausibly
    // fast for the same fixed window. Held fixed for the rest of this
    // blend so alpha stays monotonic even though target keeps moving.
    if (release_blend_duration_s_ < 0.0) {
      double total_jump_m =
          (out.translation() - release_blend_start_pose_.translation())
              .norm();
      release_blend_duration_s_ =
          std::max(kReleaseBlendDurationS, total_jump_m / kMaxBlendSpeedMps);
    }
    double elapsed_s = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() -
                            release_blend_start_wall_)
                            .count();
    if (elapsed_s >= release_blend_duration_s_) {
      release_blending_ = false;
    } else {
      double alpha =
          std::clamp(elapsed_s / release_blend_duration_s_, 0.0, 1.0);
      Sophus::SE3d delta = release_blend_start_pose_.inverse() * out;
      out = release_blend_start_pose_ * Sophus::SE3d::exp(alpha * delta.log());
    }
  }

  // Cache for checkDriftGate() to freeze onto if it trips before the
  // next call -- see that member's comment in the header for why this
  // matters (freezing at a stale detection anchor instead of "wherever
  // the live pose actually was" was a real bug found on a live test).
  last_published_pose_ = out;
  last_raw_pose_seen_ = current_raw_pose;
  have_last_published_pose_ = true;

  return true;
}

Eigen::aligned_vector<Eigen::Vector3d> OnlineLoopClosure::buildPointCloud()
    const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  Eigen::aligned_vector<Eigen::Vector3d> out;
  for (const auto& kf : keyframes_) {
    Sophus::SE3d T_w_i(composeYPR(kf.roll, kf.pitch, kf.yaw), kf.t_opt);
    Sophus::SE3d T_w_c0 = T_w_i * calib_.T_i_c[0];
    for (const auto& p_c0 : kf.pts3d) out.push_back(T_w_c0 * p_c0);
  }
  return out;
}

}  // namespace basalt
