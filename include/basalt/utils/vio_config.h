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

#include <string>
#include <vector>

namespace basalt {

enum class LinearizationType { ABS_QR, ABS_SC, REL_SC };

struct VioConfig {
  VioConfig();
  void load(const std::string& filename);
  void save(const std::string& filename);

  std::string optical_flow_type;
  int optical_flow_detection_grid_size;
  float optical_flow_max_recovered_dist2;
  int optical_flow_pattern;
  int optical_flow_max_iterations;
  int optical_flow_levels;
  float optical_flow_epipolar_error;
  int optical_flow_skip_frames;
  // Depth hypotheses (meters) tried when seeding the cam0->cam1 stereo
  // KLT track for newly-detected corners -- see
  // trackNewPointsStereoWithDepthSeeds() in frame_to_frame_optical_flow.h.
  // Configurable (rather than a fixed in-code constant) specifically so
  // a compute-constrained platform (e.g. a Pi5, vs. a laptop) can run
  // fewer hypotheses without changing the default for anyone else --
  // see conversation, 2026-09-21: a Pi5 live test showed erratic
  // per-frame tracked-point counts consistent with the combined compute
  // cost of a wide depth-hypothesis set plus higher corner density
  // overloading its 4 cores, unlike the laptop this was validated on.
  std::vector<double> optical_flow_stereo_seed_depths_m;

  LinearizationType vio_linearization_type;
  bool vio_sqrt_marg;

  int vio_max_states;
  int vio_max_kfs;
  int vio_min_frames_after_kf;
  float vio_new_kf_keypoints_thresh;
  bool vio_debug;
  bool vio_extended_logging;

  //  double vio_outlier_threshold;
  //  int vio_filter_iteration;
  int vio_max_iterations;

  double vio_obs_std_dev;
  double vio_obs_huber_thresh;
  double vio_min_triangulation_dist;

  // Static-initialization gravity/orientation estimate: instead of a single
  // accelerometer sample (fragile against handling motion right as the
  // pipeline starts -- a real, observed cause of ~0.5m position drift
  // within the first few seconds of a live run), average accel samples
  // over this many seconds leading up to the first vision frame.
  double vio_static_init_window_s;
  // If the accelerometer magnitude's stddev over that window exceeds this
  // (m/s^2), the device likely wasn't actually still -- logged as a
  // warning (init still proceeds; this isn't a hard gate) so a bad startup
  // is visible instead of silently baked into an early, uncorrected drift.
  double vio_static_init_max_accel_std;

  bool vio_enforce_realtime;

  bool vio_use_lm;
  double vio_lm_lambda_initial;
  double vio_lm_lambda_min;
  double vio_lm_lambda_max;

  bool vio_scale_jacobian;

  double vio_init_pose_weight;
  double vio_init_ba_weight;
  double vio_init_bg_weight;

  bool vio_marg_lost_landmarks;
  double vio_kf_marg_feature_ratio;

  double mapper_obs_std_dev;
  double mapper_obs_huber_thresh;
  int mapper_detection_num_points;
  double mapper_num_frames_to_match;
  double mapper_frames_to_match_threshold;
  double mapper_min_matches;
  double mapper_ransac_threshold;
  double mapper_min_track_length;
  double mapper_max_hamming_distance;
  double mapper_second_best_test_ratio;
  int mapper_bow_num_bits;
  double mapper_min_triangulation_dist;
  bool mapper_no_factor_weights;
  bool mapper_use_factors;

  bool mapper_use_lm;
  double mapper_lm_lambda_min;
  double mapper_lm_lambda_max;
};

}  // namespace basalt
