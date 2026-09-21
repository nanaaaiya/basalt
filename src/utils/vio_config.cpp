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

#include <basalt/utils/assert.h>
#include <basalt/utils/vio_config.h>

#include <fstream>

#include <cereal/archives/json.hpp>
#include <cereal/cereal.hpp>
#include <magic_enum/magic_enum.hpp>

namespace basalt {

VioConfig::VioConfig() {
  // optical_flow_type = "patch";
  optical_flow_type = "frame_to_frame";
  // Was 50 (upstream default). At this rig's 640x480 resolution that's
  // only ~108 grid cells, capping total_observed_count at roughly that
  // regardless of how much real texture is available -- directly limits
  // how many raw candidate corners exist for both the stereo-pairing and
  // temporal-triangulation paths feeding tracked_ratio (see conversation:
  // both paths already run below their own success-rate ceiling, so more
  // raw candidates should translate close to linearly into more landmarks
  // even without improving either path's percentage yield). Lowered to
  // ~2x the cell density; see also num_points_cell in
  // FrameToFrameOpticalFlow::addPoints()'s detectKeypoints() call, raised
  // from 1 to 2 for the same reason.
  optical_flow_detection_grid_size = 35;
  optical_flow_max_recovered_dist2 = 0.09f;
  optical_flow_pattern = 51;
  optical_flow_max_iterations = 5;
  optical_flow_levels = 3;
  optical_flow_epipolar_error = 0.005;
  optical_flow_skip_frames = 1;

  vio_linearization_type = LinearizationType::ABS_QR;
  vio_sqrt_marg = true;

  vio_max_states = 3;
  vio_max_kfs = 7;
  // Was 5 -- the other lever behind the post-rotation recovery lag (see
  // conversation, and kSeedDepthsM's comment in
  // frame_to_frame_optical_flow.h for the first one): once tracking
  // degrades below vio_new_kf_keypoints_thresh, this cooldown is what
  // paces how often the system even ATTEMPTS to rebuild the landmark
  // pool via a fresh keyframe. Measured live (multiple runs this
  // session) at 5, that cadence was a fixed ~7 frames / ~0.47s between
  // keyframes -- lowered to let it attempt more rebuild cycles per
  // second specifically while struggling, without changing anything
  // once tracked_ratio climbs back above threshold and keyframing goes
  // back to being scene-driven.
  vio_min_frames_after_kf = 3;
  // Was 0.7 (upstream default, tuned against clean EuRoC benchmark footage).
  // On the OAK-D Lite rig, real tracked_ratio chronically sits in the
  // 0.0-0.2 range even on well-tracked frames, so 0.7 was never satisfied
  // and this threshold alone (not real scene need) paced keyframe creation
  // at the fastest allowed rate (vio_min_frames_after_kf cooldown) for an
  // entire 205s live run -- confirmed via run 20260921_103601, where all
  // 432 keyframes were spaced exactly 7 frames apart with zero variance.
  // That churn kept re-hosting landmarks before frame-to-frame tracking
  // could build up connected observations, which fed back into keeping
  // tracked_ratio low -- triggering the starvation drift-gate trigger for
  // ~90s of that 205s run. Lowered to reflect what this rig actually
  // achieves, so new keyframes are driven by genuine scene need again.
  vio_new_kf_keypoints_thresh = 0.3;

  vio_debug = false;
  vio_extended_logging = false;
  vio_obs_std_dev = 0.5;
  vio_obs_huber_thresh = 1.0;
  vio_min_triangulation_dist = 0.05;
  vio_static_init_window_s = 0.3;
  vio_static_init_max_accel_std = 0.2;
  //  vio_outlier_threshold = 3.0;
  //  vio_filter_iteration = 4;
  vio_max_iterations = 7;

  vio_enforce_realtime = false;

  vio_use_lm = false;
  vio_lm_lambda_initial = 1e-4;
  vio_lm_lambda_min = 1e-6;
  vio_lm_lambda_max = 1e2;

  vio_scale_jacobian = true;

  vio_init_pose_weight = 1e8;
  vio_init_ba_weight = 1e1;
  vio_init_bg_weight = 1e2;

  vio_marg_lost_landmarks = true;

  vio_kf_marg_feature_ratio = 0.1;

  mapper_obs_std_dev = 0.25;
  mapper_obs_huber_thresh = 1.5;
  mapper_detection_num_points = 800;
  mapper_num_frames_to_match = 30;
  mapper_frames_to_match_threshold = 0.04;
  mapper_min_matches = 20;
  mapper_ransac_threshold = 5e-5;
  mapper_min_track_length = 5;
  mapper_max_hamming_distance = 70;
  mapper_second_best_test_ratio = 1.2;
  mapper_bow_num_bits = 16;
  mapper_min_triangulation_dist = 0.07;
  mapper_no_factor_weights = false;
  mapper_use_factors = true;

  mapper_use_lm = false;
  mapper_lm_lambda_min = 1e-32;
  mapper_lm_lambda_max = 1e2;
}

void VioConfig::save(const std::string& filename) {
  std::ofstream os(filename);

  {
    cereal::JSONOutputArchive archive(os);
    archive(*this);
  }
  os.close();
}

void VioConfig::load(const std::string& filename) {
  std::ifstream is(filename);

  {
    cereal::JSONInputArchive archive(is);
    archive(*this);
  }
  is.close();
}
}  // namespace basalt

namespace cereal {

template <class Archive>
std::string save_minimal(const Archive& ar,
                         const basalt::LinearizationType& linearization_type) {
  UNUSED(ar);
  auto name = magic_enum::enum_name(linearization_type);
  return std::string(name);
}

template <class Archive>
void load_minimal(const Archive& ar,
                  basalt::LinearizationType& linearization_type,
                  const std::string& name) {
  UNUSED(ar);

  auto lin_enum = magic_enum::enum_cast<basalt::LinearizationType>(name);

  if (lin_enum.has_value()) {
    linearization_type = lin_enum.value();
  } else {
    std::cerr << "Could not find the LinearizationType for " << name
              << std::endl;
    std::abort();
  }
}

template <class Archive>
void serialize(Archive& ar, basalt::VioConfig& config) {
  ar(CEREAL_NVP(config.optical_flow_type));
  ar(CEREAL_NVP(config.optical_flow_detection_grid_size));
  ar(CEREAL_NVP(config.optical_flow_max_recovered_dist2));
  ar(CEREAL_NVP(config.optical_flow_pattern));
  ar(CEREAL_NVP(config.optical_flow_max_iterations));
  ar(CEREAL_NVP(config.optical_flow_epipolar_error));
  ar(CEREAL_NVP(config.optical_flow_levels));
  ar(CEREAL_NVP(config.optical_flow_skip_frames));

  ar(CEREAL_NVP(config.vio_linearization_type));
  ar(CEREAL_NVP(config.vio_sqrt_marg));
  ar(CEREAL_NVP(config.vio_max_states));
  ar(CEREAL_NVP(config.vio_max_kfs));
  ar(CEREAL_NVP(config.vio_min_frames_after_kf));
  ar(CEREAL_NVP(config.vio_new_kf_keypoints_thresh));
  ar(CEREAL_NVP(config.vio_debug));
  ar(CEREAL_NVP(config.vio_extended_logging));
  ar(CEREAL_NVP(config.vio_max_iterations));
  //  ar(CEREAL_NVP(config.vio_outlier_threshold));
  //  ar(CEREAL_NVP(config.vio_filter_iteration));

  ar(CEREAL_NVP(config.vio_obs_std_dev));
  ar(CEREAL_NVP(config.vio_obs_huber_thresh));
  ar(CEREAL_NVP(config.vio_min_triangulation_dist));
  ar(CEREAL_NVP(config.vio_static_init_window_s));
  ar(CEREAL_NVP(config.vio_static_init_max_accel_std));

  ar(CEREAL_NVP(config.vio_enforce_realtime));

  ar(CEREAL_NVP(config.vio_use_lm));
  ar(CEREAL_NVP(config.vio_lm_lambda_initial));
  ar(CEREAL_NVP(config.vio_lm_lambda_min));
  ar(CEREAL_NVP(config.vio_lm_lambda_max));

  ar(CEREAL_NVP(config.vio_scale_jacobian));

  ar(CEREAL_NVP(config.vio_init_pose_weight));
  ar(CEREAL_NVP(config.vio_init_ba_weight));
  ar(CEREAL_NVP(config.vio_init_bg_weight));

  ar(CEREAL_NVP(config.vio_marg_lost_landmarks));
  ar(CEREAL_NVP(config.vio_kf_marg_feature_ratio));

  ar(CEREAL_NVP(config.mapper_obs_std_dev));
  ar(CEREAL_NVP(config.mapper_obs_huber_thresh));
  ar(CEREAL_NVP(config.mapper_detection_num_points));
  ar(CEREAL_NVP(config.mapper_num_frames_to_match));
  ar(CEREAL_NVP(config.mapper_frames_to_match_threshold));
  ar(CEREAL_NVP(config.mapper_min_matches));
  ar(CEREAL_NVP(config.mapper_ransac_threshold));
  ar(CEREAL_NVP(config.mapper_min_track_length));
  ar(CEREAL_NVP(config.mapper_max_hamming_distance));
  ar(CEREAL_NVP(config.mapper_second_best_test_ratio));
  ar(CEREAL_NVP(config.mapper_bow_num_bits));
  ar(CEREAL_NVP(config.mapper_min_triangulation_dist));
  ar(CEREAL_NVP(config.mapper_no_factor_weights));
  ar(CEREAL_NVP(config.mapper_use_factors));

  ar(CEREAL_NVP(config.mapper_use_lm));
  ar(CEREAL_NVP(config.mapper_lm_lambda_min));
  ar(CEREAL_NVP(config.mapper_lm_lambda_max));
}
}  // namespace cereal
