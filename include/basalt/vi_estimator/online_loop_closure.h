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
//
// KNOWN LIMITATION (accepted, not fixed): a single wrong long-range loop
// closure can meaningfully corrupt the corrected trajectory, and this
// pose-only graph has no reliable way to reject one in advance. Root-
// caused on an EuRoC dataset test (V1_01_easy, a small Vicon capture room
// covered in repetitive calibration markers -- a "perceptual aliasing"
// setup): a match ~36s in the past with only 28 supporting points got
// accepted and pulled one keyframe's position badly off, and that error
// then propagated forward through the normal keyframe chain. Three fixes
// were tried:
//   1. Minimum time gap before a candidate is eligible (kept, real,
//      insufficient alone -- see kMinLoopClosureTimeGapNs below).
//   2. Requiring more supporting points for a longer time gap (tried,
//      REVERTED -- made results measurably worse, because a genuine
//      distant revisit also naturally has fewer points than a recent one,
//      since more time means more viewpoint/lighting change even when the
//      match is correct; inlier count alone can't separate "far and
//      right" from "far and wrong").
//   3. Robust (Huber) down-weighting of loop edges in the solve (kept, no
//      measurable improvement on the diagnosed case -- see
//      kLoopEdgeHuberDeltaM below). Doesn't help here because the
//      corrupted node only has 3 edges touching it (2 odometry + the bad
//      loop edge); there's no independent evidence in this thin pose-only
//      graph to outvote a bad edge with, so the solver just satisfies all
//      three edges by bending to the wrong position instead of the bad
//      edge showing up as a residual outlier.
//   4. Accepting more than one independently-verified closure per
//      keyframe (up to kMaxLoopEdgesPerKeyframe, .cpp), instead of
//      stopping at the first -- targets exactly the "corrupted node had
//      only 3 edges" gap #3 describes, by giving the solver a real
//      chance at competing evidence from the start rather than relying
//      on some later, unrelated closure to happen to touch the same
//      node. VERIFIED via scripts/eval_full/run_loop_closure_eval.sh on
//      V1_01_easy (same build, same commit, only kMaxLoopEdgesPerKeyframe
//      changed) -- but note the pipeline is NOT deterministic run-to-run
//      (RANSAC randomization and/or TBB parallel-reduce floating-point
//      order both plausible causes: re-running the *identical* binary on
//      the *identical* input produced corrected_ate_rmse ranging
//      1.296-1.392m across 3 runs), so a single-run comparison would have
//      been misleading. Across 3 runs each: 2 edges -> corrected_ate_rmse
//      {1.296, 1.392, 1.303}m (mean 1.330m), 178/175/176 closures; 1 edge
//      -> {1.341, 1.388, 1.468}m (mean 1.399m), 114/112/111 closures.
//      Direction is consistent (2 edges beat 1 edge's mean in all 3
//      samples) but the ~5% effect size is close to the ~5% run-to-run
//      noise band, so treat this as a small, directionally-consistent,
//      NOT strongly statistically confident improvement -- not a crisp
//      fixed percentage. Either way, it does NOT fix this failure mode:
//      every single sample from both configurations remains over 25x
//      worse than raw VIO (0.043m) on this exact repro case. Do not
//      treat multi-edge redundancy as a solution to perceptual aliasing;
//      it is at most a modest mitigation. This result (and the noise
//      characterization itself) is the actual merge-readiness evidence
//      for the loop-closure-multi-edge-redundancy branch -- if tighter
//      confidence is needed before merging, re-run with a larger sample
//      per configuration rather than trusting either single number.
// The real fix would be a richer graph with actual landmark-level
// redundancy (multiple independent point observations per closure, like
// Basalt's own offline basalt_mapper) -- deliberately NOT pursued now:
// that's a major redesign (comparable in scope to this module), carries
// real risk of new scaling problems (this session already found three
// unrelated ones in the current, much simpler design as it scaled up),
// and isn't proven to fully fix aliasing even then. Every live OAK-D test
// this session (the actual deployment target) has performed well with no
// sign of this failure mode -- this was found via an EuRoC scaling side-
// investigation, on an unusually repetitive dataset, not the real use
// case. Revisit toward a richer graph only if a REAL deployment test
// shows this same failure mode in a genuinely repetitive real-world
// environment, not preemptively.

#pragma once

#include <atomic>
#include <chrono>
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

// See OnlineLoopClosure::checkDriftGate() (.cpp) for what triggers each of
// these. The two release cases look identical from outside if only a
// bool is reported -- kReleasedConfirmed means 3 consecutive solves
// actually agreed with raw-chained motion again, kReleasedForced means
// the kDriftGateMaxHoldSeconds timeout gave up waiting and accepted
// whatever the graph currently says, with no confirmation at all. A
// consumer judging how much to trust the pose right after a release
// needs that distinction -- "confirmed" and "forced" carry very
// different reliability even though both end the hold.
enum class DriftGateEvent { kTripped, kReleasedConfirmed, kReleasedForced };

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

  // Pushed every time the drift gate's held state actually CHANGES -- a
  // consumer polling isDriftHeld() at a fixed interval (e.g. for a
  // dashboard alert) can otherwise miss rapid trip/release cycles that
  // happen faster than its poll rate, confirmed on a real live test where
  // several undercounted cycles made the gate look like one continuous,
  // unexplained freeze instead of the flapping it actually was. Drain
  // this instead of (or alongside) polling isDriftHeld() to report every
  // transition faithfully. See DriftGateEvent above for what each value
  // means, in particular the confirmed-vs-forced release distinction.
  // mutable: pushed to from forceReleaseDriftHoldLocked(), which is const
  // (see its declaration) so it's callable from getSmoothedCorrectedPose().
  mutable tbb::concurrent_bounded_queue<DriftGateEvent> drift_gate_events;

  void start();
  void stop();

  // Thread-safe snapshot of the latest globally-corrected trajectory,
  // ordered by keyframe arrival (== time order).
  Eigen::aligned_vector<Eigen::Vector3d> getCorrectedTrajectory() const;

  // Returns the newest pose-graph node's own position directly. Real, but
  // architecturally the LEAST stable pose to show live: the newest node
  // has the fewest accumulated edges of any node in the graph, so its own
  // position shifts the most every time the graph gets re-solved (i.e. on
  // every new loop closure). With a stationary or slow-moving camera in a
  // visually repetitive scene, closures can fire almost continuously
  // (confirmed live: 197 closures over 325 keyframes in one near-
  // stationary Pi5 test), and each re-solve's nudge to this node reads as
  // a visible jump with no real motion to absorb it. Prefer
  // getSmoothedCorrectedPose() for anything drawn/streamed live; this one
  // remains for existing offline/dataset callers (see vio.cpp) not yet
  // migrated.
  bool getLatestCorrectedPose(Sophus::SE3d& out) const;

  // Live-display alternative to getLatestCorrectedPose() that doesn't
  // inherit the newest node's volatility: rebases the latest corrected
  // KEYFRAME's pose forward by the raw VIO motion since that keyframe
  // (T_corrected_kf * T_kf_raw^-1 * current_raw_pose), rather than
  // exposing the newest node's own position directly. This updates
  // smoothly every call by following real, high-rate raw motion, and only
  // ever "steps" when the reference keyframe itself changes or gets a
  // fresh correction -- the same loop-closure-snap behavior already
  // expected elsewhere in this codebase (see oak_d_vio.cpp's
  // camera-follow comment), just no longer amplified by re-solve noise on
  // top of it. Returns false under the same condition as
  // getLatestCorrectedPose() (no keyframes yet).
  bool getSmoothedCorrectedPose(const Sophus::SE3d& current_raw_pose,
                                 Sophus::SE3d& out) const;

  // True while the drift gate is holding the live pose (see
  // solvePoseGraph()'s drift-gate comment and kDriftGateThresholdM in the
  // .cpp) -- getSmoothedCorrectedPose() freezes at the last trusted pose
  // while this is true, instead of advancing onto a newly-solved position
  // whose odometry-edge residual looks implausible. Exposed so a caller
  // (e.g. oak_d_vio.cpp) can surface a clear alert when this changes,
  // rather than the pose just silently stopping.
  bool isDriftHeld() const;

  // True for kForcedReleaseCooldownS after a FORCED drift-gate release
  // (the max-hold timeout gave up waiting, not a confirmed-good recovery
  // -- see DriftGateEvent and last_forced_release_wall_'s comment). Lets
  // computeVioConfidence() report reduced confidence for a pose that was
  // just resynced without confirmation, instead of the derived score
  // snapping straight back to "nominal" the instant the hold ends.
  bool isRecentlyForceReleased() const;

  // Feeds the same tracked_ratio/total_observed_count oak_d_vio.cpp
  // already reads for VioConfidenceInputs into the drift gate's
  // starvation trigger (see kStarvationTrackedRatioThresh in the .cpp) --
  // call this once per VIO frame from wherever that health block already
  // lives. Deliberately a separate "push" method rather than adding
  // parameters to getSmoothedCorrectedPose() itself: that method has six
  // call sites across oak_d_vio.cpp, and not all of them are positioned
  // to easily supply a fresh value on every call, whereas the health
  // block already computes these once per frame in one place. Cheap
  // (two atomic writes), safe to call even when online loop closure is
  // otherwise idle.
  void reportTrackingHealth(double tracked_ratio, int total_observed_count);

  // Same as getCorrectedTrajectory(), but paired with each keyframe's
  // timestamp -- needed for logging/analysis (matching timestamps up
  // against the raw VIO trajectory, sample rate, etc.), not just drawing a
  // line in the GUI.
  void getCorrectedTrajectoryWithTimestamps(
      std::vector<int64_t>& t_ns,
      Eigen::aligned_vector<Eigen::Vector3d>& positions) const;

  int numLoopClosures() const { return num_loop_closures.load(); }

  // Stereo-triangulated point count from the most recently processed
  // keyframe (already printed every keyframe as [STEREO-DIAG]
  // "triangulated=", now also queryable) -- a confidence signal
  // (vio_health.h) / scenario-characterization input.
  int getLatestTriangulatedPoints() const {
    return latest_triangulated_points.load();
  }

  // Every stereo-triangulated point from every keyframe, transformed into
  // the corrected world frame -- for live map-saving (see
  // DashboardClient::sendMapFile()). Deliberately NOT routed through the
  // offline basalt_mapper/--marg-data pipeline: this uses data already
  // held here for live correction, so a map is available immediately,
  // mid-flight, without needing marg-data or a separate process. Unlike
  // getCorrectedTrajectory() (one point per keyframe -- its position),
  // this returns every landmark, so it can be much larger, and has no
  // landmark deduplication or bundle adjustment -- that's what the
  // offline basalt_mapper (--marg-data, --save-map) still does better,
  // whenever a higher-quality post-flight map is wanted instead of an
  // immediate live one.
  Eigen::aligned_vector<Eigen::Vector3d> buildPointCloud() const;

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

    // True for loop-closure edges, false for sequential odometry edges.
    // Only loop edges get robust (Huber) down-weighting in the solve --
    // see solvePoseGraph() -- since the odometry chain is the trusted
    // backbone that should resist being dragged by a bad loop edge, not
    // get weakened alongside it.
    bool is_loop = false;
  };

  void processingLoop();
  void processKeyframe(const MargData::Ptr& data, int64_t kf_id);
  void solvePoseGraph();
  void checkDriftGate();  // called by solvePoseGraph() -- see its .cpp comment

  // Shared by checkDriftGate()'s keyframe-gated timeout and
  // getSmoothedCorrectedPose()'s wall-clock watchdog (see
  // drift_held_since_wall_) -- both need to trigger the exact same
  // force-release bookkeeping, just from different call paths. Assumes
  // state_mutex_ is already held by the caller (const so it's callable
  // from getSmoothedCorrectedPose() too; the mutable members it writes
  // are the reason those are mutable).
  void forceReleaseDriftHoldLocked() const;

  Calibration<double> calib_;
  VioConfig config_;

  std::unique_ptr<HashBow<256>> hash_bow_;

  std::vector<LoopKeyframe> keyframes_;  // arrival order == time order
  std::unordered_map<int64_t, size_t> t_ns_to_idx_;
  std::vector<PoseGraphEdge> edges_;

  // Timestamp of the very first keyframe this instance ever processed,
  // captured unconditionally (bypassing the normal quality gate) as a
  // dedicated "home" reference for a future takeoff-point revisit check --
  // not consumed anywhere yet, but costs nothing to track now and avoids
  // the alternative of only having a home reference if the first keyframe
  // happens to also clear the quality bar (plausible to fail right at
  // startup, camera still settling).
  int64_t home_keyframe_t_ns_ = -1;

  // Drift gate (see checkDriftGate() in the .cpp): while held, the live
  // pose (getSmoothedCorrectedPose()) freezes at held_pose_ instead of
  // advancing, and the graph keeps solving normally in the background so
  // it still has a chance to self-correct before the hold is released.
  //
  // held_pose_/held_anchor_raw_pose_ are seeded from last_published_pose_/
  // last_raw_pose_seen_ at the moment of tripping -- i.e. wherever the
  // live pose actually was an instant ago -- NOT from drift_anchor_idx_'s
  // node. A real live test caught this: the detection anchor only
  // refreshes every kDriftGateAnchorRefreshKeyframes solve events, which
  // can span far more real time/keyframes than that if closures are
  // sparse, so it can still be sitting at keyframe 0 (the VIO world
  // origin) when a trip happens well into a flight -- freezing there
  // teleports the live pose back to (0,0,0) instead of holding it in
  // place, which is a worse failure than the drift it's meant to guard
  // against.
  // mutable: forceReleaseDriftHoldLocked() is called from
  // getSmoothedCorrectedPose() (const, protected by state_mutex_ like
  // every other method here) as well as checkDriftGate() (non-const) --
  // see that method's comment for why a release needs to be triggerable
  // from both places.
  mutable bool drift_held_ = false;
  // mutable: both written from checkDriftGate() (non-const) on a normal
  // residual-based trip, AND from getSmoothedCorrectedPose() (const) on
  // a starvation trip -- see kStarvationTrackedRatioThresh in the .cpp.
  mutable Sophus::SE3d held_pose_;
  mutable Sophus::SE3d held_anchor_raw_pose_;
  mutable int drift_gate_stable_count_ = 0;
  // t_ns the current hold started at -- see kDriftGateMaxHoldSeconds:
  // a real live test found this staying stuck (never confirmed
  // recovered) for 40+ seconds straight, which is worse for a live
  // display than resuming with an unconfirmed correction. Only checked
  // from checkDriftGate(), which runs on keyframe arrival -- see
  // drift_held_since_wall_ for the gap that left. Set to -1 (meaning
  // "no keyframe-time anchor, use the wall-clock twin instead") by a
  // starvation trip, which by definition isn't backed by a fresh
  // keyframe -- mutable for the same reason as held_pose_ above.
  mutable int64_t drift_held_since_t_ns_ = -1;

  // Wall-clock twin of drift_held_since_t_ns_, checked from
  // getSmoothedCorrectedPose() instead of checkDriftGate(). Necessary
  // because checkDriftGate() only runs when solvePoseGraph() processes a
  // new keyframe -- confirmed on a real live test: tracking got bad
  // enough that no keyframe was processed for 33+ seconds, during which
  // the keyframe-gated timeout above simply never got a chance to run,
  // so a hold meant to cap at 15s instead ran for 39.79s. This is
  // checked from getSmoothedCorrectedPose() specifically because that's
  // called at full pose-publish rate regardless of keyframe activity,
  // so the max-hold promise holds even when nothing else is running.
  mutable std::chrono::steady_clock::time_point drift_held_since_wall_;

  // When a hold releases (either way -- see checkDriftGate()), the very
  // next getSmoothedCorrectedPose() call would otherwise jump straight
  // from held_pose_ to wherever the graph currently says, in one step --
  // a real live test observed this as a visible "teleport" on the GUI,
  // confirmed to coincide with a release event down to the second. This
  // doesn't make the destination any more correct (a bad correction
  // glided-to is still bad), but it removes the discontinuity itself,
  // which matters beyond cosmetics: a downstream consumer (e.g. a flight
  // controller) seeing an instantaneous, physically-impossible position
  // jump could react to it as if it were real motion. Timed off wall
  // clock rather than threaded through every getSmoothedCorrectedPose()
  // call site (there are several) with a VIO timestamp -- this is a
  // fixed real-world duration for display/consumer smoothness, not tied
  // to VIO's own logical time.
  mutable bool release_blending_ = false;
  mutable Sophus::SE3d release_blend_start_pose_;
  mutable std::chrono::steady_clock::time_point release_blend_start_wall_;
  static constexpr double kReleaseBlendDurationS = 0.4;
  // Actual duration in use for the blend currently in flight -- a fixed
  // 0.4s is fine for a small correction but, measured live (run
  // 20260921_114840), a large one (20-40cm) spread over just 0.4s still
  // produced individual published-tick deltas over 5cm, since a bigger
  // total step divided by the same fixed window is a bigger per-tick
  // slice. Lazily computed the first time getSmoothedCorrectedPose()
  // applies a newly-armed blend (every arm site just sets this to -1;
  // whichever call notices release_blending_ went true first fills it
  // in using the actual start/target delta it can see, rather than
  // duplicating that computation at every arm site), then held fixed
  // for the rest of that blend so alpha stays monotonic. See
  // kMaxBlendSpeedMps for how it's derived.
  mutable double release_blend_duration_s_ = -1.0;
  // Caps how fast a blended correction is allowed to visibly move --
  // chosen near the upper end of plausible handheld/light-drone motion
  // seen in this session's own data (most measured live jumps implied
  // well under 1 m/s), so a blended correction never looks like faster,
  // more violent motion than the vehicle could plausibly be doing.
  static constexpr double kMaxBlendSpeedMps = 0.5;

  // Ordinary target movement, independent of the drift gate --
  // getSmoothedCorrectedPose()'s normal (not-held) path anchors on
  // keyframes_.back() and chains raw motion forward from it. The
  // resulting unblended target can jump for two DIFFERENT reasons, not
  // just one: (1) the anchor switches identity to a freshly-created
  // keyframe, or (2) solvePoseGraph() -- run whenever a loop closure is
  // accepted -- revises the CURRENT anchor's t_opt/yaw IN PLACE (it
  // writes back every free node's solved position each time it runs,
  // not just a newly-added one), with no identity change at all. An
  // earlier version of this fix only watched for (1) via a keyframe-id
  // comparison and completely missed (2) -- confirmed live (run
  // 20260921_115907) by unblended jumps up to 22.5cm/tick (~3.4 m/s)
  // that didn't correlate with any anchor-identity change or drift-gate
  // event. Comparing the freshly-computed unblended target directly
  // against the last unblended target (not the possibly-still-blending
  // last_published_pose_, which is deliberately lagging behind target
  // while a blend is in flight and would falsely look like continuous
  // movement) catches both cases uniformly, and normal frame-to-frame
  // raw motion stays well under the threshold below so it doesn't
  // spuriously re-arm.
  mutable Sophus::SE3d last_unblended_target_pose_;
  mutable bool have_last_unblended_target_pose_ = false;
  // Below this, don't bother starting a blend -- ordinary graph-solve
  // noise between consecutive publishes is usually a few mm to low-cm
  // and blending on every single one adds pointless state churn for
  // something already invisible against the <5cm accuracy target.
  static constexpr double kTargetJumpBlendThresholdM = 0.02;

  // Set on a FORCED release specifically (kDriftGateMaxHoldSeconds timeout
  // gave up waiting, not confirmed by kDriftGateReleaseCount consecutive
  // good solves -- see DriftGateEvent) and read by
  // isRecentlyForceReleased() for a short cooldown afterward, so
  // computeVioConfidence() can report reduced confidence for a pose that
  // was just resynced without confirmation instead of snapping straight
  // back to "nominal".
  mutable std::chrono::steady_clock::time_point last_forced_release_wall_;
  mutable bool had_forced_release_ = false;
  static constexpr double kForcedReleaseCooldownS = 5.0;

  // Starvation trigger state (see kStarvationTrackedRatioThresh in the
  // .cpp for the full reasoning) -- a second way into drift_held_,
  // independent of checkDriftGate()'s residual check. Written by
  // reportTrackingHealth() (called once per VIO frame from
  // oak_d_vio.cpp), read from getSmoothedCorrectedPose(). Defaults to
  // healthy values so nothing trips before the first real report
  // arrives (e.g. during startup, before VIO has produced any health
  // reading at all).
  std::atomic<double> latest_reported_tracked_ratio_{1.0};
  std::atomic<int> latest_reported_total_observed_count_{999};
  // Wall-clock time the CURRENT continuous starvation stretch began --
  // reset to invalid (via starvation_active_) the moment the condition
  // stops holding, same "wall-clock, not a call count" reasoning as
  // kStarvationPersistenceSeconds's own comment.
  mutable bool starvation_active_ = false;
  mutable std::chrono::steady_clock::time_point starvation_since_wall_;

  // Continuously updated by getSmoothedCorrectedPose() (mutable: that
  // method is const) every time it publishes a live, not-held pose --
  // this is what checkDriftGate() actually freezes onto, per the comment
  // above.
  mutable Sophus::SE3d last_published_pose_;
  mutable Sophus::SE3d last_raw_pose_seen_;
  mutable bool have_last_published_pose_ = false;

  // The FIXED reference node used for detecting a NEW trip (separate
  // from held_pose_/held_anchor_raw_pose_, which only matter once
  // already held). Deliberately not recomputed as "N keyframes back"
  // every check -- see kDriftGateAnchorRefreshKeyframes's comment for
  // why that let a slow, real accumulation go undetected on a live test.
  mutable size_t drift_anchor_idx_ = 0;
  mutable size_t keyframes_since_anchor_refresh_ = 0;

  mutable std::mutex state_mutex_;
  // Counts accepted loop EDGES, not keyframes that found a match -- a
  // single keyframe can now contribute more than one (see
  // kMaxLoopEdgesPerKeyframe in the .cpp), so this can exceed the number
  // of keyframes that ever closed a loop.
  std::atomic<int> num_loop_closures{0};
  std::atomic<int> latest_triangulated_points{0};

  std::atomic<bool> running{false};
  std::thread worker_thread_;
};

}  // namespace basalt
