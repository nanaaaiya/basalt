/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2019, Vladyslav Usenko, Michael Loipführer and Nikolaus Demmel.
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

// Live VIO app for the Luxonis OAK-D Lite, modeled directly on
// rs_t265_vio.cpp. Unlike the T265 (which can export its own factory
// calibration), the OAK-D always requires an external --cam-calib file --
// see results/calibration_final.json, produced via Basalt's own calibration
// tools seeded/patched with known-good Kalibr values for this exact unit.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <ctime>
#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

#include <sophus/se3.hpp>

#include <opencv2/imgcodecs.hpp>

#include <basalt/utils/thread_priority.h>

#include <tbb/concurrent_queue.h>
#include <tbb/global_control.h>

#include <pangolin/display/default_font.h>
#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/pangolin.h>

#include <CLI/CLI.hpp>

#include <basalt/device/oak_d.h>
#include <basalt/io/dashboard_client.h>
#include <basalt/io/dataset_io.h>
#include <basalt/io/marg_data_io.h>
#include <basalt/mapping/occupancy_mapper.h>
#include <basalt/utils/filesystem.h>
#include <basalt/utils/vio_health.h>
#include <basalt/spline/se3_spline.h>
#include <basalt/vi_estimator/online_loop_closure.h>
#include <basalt/vi_estimator/vio_estimator.h>
#include <basalt/calibration/calibration.hpp>

#include <basalt/serialization/headers_serialization.h>

#include <basalt/utils/vis_utils.h>

// GUI functions
void draw_image_overlay(pangolin::View& v, size_t cam_id);
void draw_scene();
void load_data(const std::string& calib_path);
void draw_plots();
void drain_vio_plot_queue();
void drain_localization_queue();
basalt::VioVisualizationData::Ptr get_curr_vis_data_snapshot();

// Saves the raw and (if enabled) loop-closure-corrected trajectories to
// disk, plus a summary with the start-to-end distance for each -- the
// quantitative replacement for eyeballing drift off the live GUI.
void write_trajectory_logs(const std::string& log_dir);

// Pangolin variables
constexpr int UI_WIDTH = 200;

basalt::OakDDevice::Ptr oakd_device;
basalt::OnlineLoopClosure::Ptr online_loop_closure;
basalt::MargDataFanOut::Ptr marg_fan_out;
basalt::DashboardClient::Ptr dashboard_client;
basalt::OccupancyMapper::Ptr occupancy_mapper;
tbb::concurrent_bounded_queue<std::shared_ptr<dai::ImgFrame>> depth_queue;

using Button = pangolin::Var<std::function<void(void)>>;

// Global (not local to main()) so the SIGINT/SIGTERM handler below can reach
// it -- a plain signal handler can only touch process-wide state. Ctrl+C
// used to kill the process instantly, skipping the shutdown block at the
// end of main() (thread joins + write_trajectory_logs()), so no trajectory
// log ever got written unless the Pangolin window was closed by hand
// instead. Setting this flag and nudging Pangolin to quit makes Ctrl+C fall
// through to that same clean-shutdown path.
std::atomic<bool> terminate{false};

void handle_shutdown_signal(int /*signum*/) {
  terminate = true;
  pangolin::QuitAll();
}

// Duplicates every byte written to it into two underlying streambufs --
// used to mirror stdout/stderr into a console.log file inside
// run_logs/<timestamp>/ while still printing live to the terminal.
// Post-run analysis needs the actual [ONLINE-LOOP] per-keyframe decision
// log, not just the trajectory numbers, so this makes that automatic
// instead of relying on remembering to add `| tee` on the command line.
// Every thread in this app (t3/t4/t6, the VIO filter thread,
// OnlineLoopClosure's thread, main) writes to std::cout/std::cerr
// concurrently and unsynchronized -- both get retargeted to instances of
// this class below, and both instances' `b_` is the SAME underlying
// console_log_file streambuf. Without a shared lock, that's a genuine
// data race on that one file's internal buffer state: observed in
// practice as a duplicated log line ("Finished t3" printed twice) and,
// on real hardware, a `corrupted double-linked list` glibc heap-
// corruption abort at shutdown. mu_ is a single mutex shared by every
// TeeStreambuf instance (static, not per-instance) so a cout-writer and
// a cerr-writer serialize against each other too, not just against
// other writers on the same stream.
class TeeStreambuf : public std::streambuf {
 public:
  TeeStreambuf(std::streambuf* a, std::streambuf* b) : a_(a), b_(b) {}

 protected:
  int overflow(int c) override {
    if (c == EOF) return !EOF;
    std::lock_guard<std::mutex> lock(mu_);
    bool ok_a = a_->sputc(static_cast<char>(c)) != EOF;
    bool ok_b = b_->sputc(static_cast<char>(c)) != EOF;
    return (ok_a && ok_b) ? c : EOF;
  }

  int sync() override {
    std::lock_guard<std::mutex> lock(mu_);
    int ra = a_->pubsync();
    int rb = b_->pubsync();
    return (ra == 0 && rb == 0) ? 0 : -1;
  }

 private:
  std::streambuf* a_;
  std::streambuf* b_;
  static std::mutex mu_;
};

std::mutex TeeStreambuf::mu_;

pangolin::DataLog imu_data_log, vio_data_log, error_data_log;
pangolin::Plotter* plotter;

pangolin::Var<bool> show_obs("ui.show_obs", true, true);
pangolin::Var<bool> show_ids("ui.show_ids", false, true);

pangolin::Var<bool> show_est_pos("ui.show_est_pos", true, true);
pangolin::Var<bool> show_est_vel("ui.show_est_vel", false, true);
pangolin::Var<bool> show_est_bg("ui.show_est_bg", false, true);
pangolin::Var<bool> show_est_ba("ui.show_est_ba", false, true);

pangolin::Var<bool> follow("ui.follow", true, true);
pangolin::Var<bool> show_raw_traj("ui.show_raw_traj", true, true);

// Visualization variables
basalt::VioVisualizationData::Ptr curr_vis_data;
std::mutex curr_vis_data_mutex;

tbb::concurrent_bounded_queue<basalt::VioVisualizationData::Ptr> out_vis_queue;
tbb::concurrent_bounded_queue<basalt::PoseVelBiasState<double>::Ptr>
    out_state_queue;

std::vector<int64_t> vio_t_ns;
Eigen::aligned_vector<Eigen::Vector3d> vio_t_w_i;

// Latest verified visual-match localization result (see LocalizationResult
// in online_loop_closure.h), drained from online_loop_closure->
// localization_queue once per render frame -- both the drain and the draw
// happen on this same single GUI thread, so no locking is needed, same as
// the vio_data_log/imu_data_log pattern below. Gives immediate feedback on
// a small "look at a known place" test without needing a full corrected-
// trajectory drift readout.
bool has_localization = false;
basalt::LocalizationResult latest_localization;

std::string marg_data_path;

bool step_by_step = false;
int64_t curr_t_ns = -1;
// Latest raw VIO pose, for the occupancy mapper to use when
// --online-loop-closure isn't active (or hasn't produced a corrected pose
// yet) -- guarded by vio_state_mutex like curr_t_ns/vio_t_w_i above.
Sophus::SE3d curr_raw_pose;
// Short time-indexed history of raw poses (ascending t_ns), so a depth
// frame can be paired with the pose that was actually true AT ITS OWN
// CAPTURE TIME (see t7's lookupRawPoseNear()), not just "whatever pose is
// freshest when the depth-processing thread gets around to it" -- under
// real camera motion those two differ by however long the frame sat in
// depth_queue plus t7's own occupancy_rate_hz throttling, and every bit of
// that gap directly smears the resulting point cloud (a stationary test
// never surfaces this -- pose doesn't change either way -- which is why
// it passed hand-verification earlier; a moving test does). Live-
// diagnosed on OAK-D Pro W, 2026-09-29: voxels stopped forming a clean
// wall-like surface specifically once the camera was actually moved
// around, not just held still. Capped (not unbounded) since this is only
// ever queried for very recent frames.
std::deque<std::pair<int64_t, Sophus::SE3d>> raw_pose_history;
constexpr size_t kRawPoseHistoryCap = 400;
std::mutex vio_state_mutex;
tbb::concurrent_bounded_queue<std::vector<float>> vio_plot_queue;

// Raw IMU tap, forwarded to opt_flow_ptr->addIMUToQueue() by t8 below --
// see OakDDevice::setImuTapQueue()'s comment for why this is separate
// from vio->imu_data_queue (the estimator's own consumption queue).
tbb::concurrent_bounded_queue<basalt::ImuData<double>::Ptr> imu_tap_queue;

// VIO variables
basalt::Calibration<double> calib;

basalt::VioConfig vio_config;
basalt::OpticalFlowBase::Ptr opt_flow_ptr;
basalt::VioEstimatorBase::Ptr vio;

int main(int argc, char** argv) {
  std::signal(SIGINT, handle_shutdown_signal);
  std::signal(SIGTERM, handle_shutdown_signal);

  bool show_gui = true;
  bool print_queue = false;
  std::string cam_calib_path;
  std::string config_path;
  int num_threads = 0;
  bool use_double = false;

  CLI::App app{"OAK-D Lite Live Vio"};

  app.add_option("--show-gui", show_gui, "Show GUI");
  app.add_option("--cam-calib", cam_calib_path, "Camera calibration file.")
      ->required();

  app.add_option("--marg-data", marg_data_path,
                 "Path to folder where marginalization data will be stored.");

  app.add_option("--print-queue", print_queue, "Print queue.");
  app.add_option("--config-path", config_path, "Path to config file.");
  app.add_option("--num-threads", num_threads, "Number of threads.");
  app.add_option("--step-by-step", step_by_step, "Path to config file.");
  app.add_option("--use-double", use_double, "Use double not float.");

  bool online_loop_closure_enabled = false;
  app.add_option("--online-loop-closure", online_loop_closure_enabled,
                 "Enable live loop-closure correction (see OnlineLoopClosure).");

  // VIO Dashboard integration (see basalt::DashboardClient). Leave
  // --dashboard-host empty to disable -- no connection is attempted and
  // every dashboard_client call site below is a no-op check.
  std::string dashboard_host;
  app.add_option("--dashboard-host", dashboard_host,
                 "VIO Dashboard backend IP. Leave empty to disable.");
  int dashboard_port = 8765;
  app.add_option("--dashboard-port", dashboard_port,
                 "Dashboard backend's PI5_TCP_PORT.");

  // Live 3D occupancy-grid mapping (see basalt::OccupancyMapper). Enables
  // the OAK-D's on-device StereoDepth node too -- see OakDDevice's
  // enable_stereo_depth constructor flag. See the "KNOWN LIMITATION" note
  // at the top of occupancy_mapper.h before relying on this for a real
  // flight -- a pre-existing OAK-D/USB instability, independent of this
  // feature, can crash the whole process under sustained real-hardware use.
  bool enable_occupancy_mapping = false;
  app.add_option("--enable-occupancy-mapping", enable_occupancy_mapping,
                 "Build a live 3D occupancy grid from stereo depth "
                 "(see OccupancyMapper).");
  double occupancy_voxel_size = 0.2;
  app.add_option("--occupancy-voxel-size", occupancy_voxel_size,
                 "Occupancy grid voxel size in meters.");
  int occupancy_depth_stride = 4;
  app.add_option("--occupancy-depth-stride", occupancy_depth_stride,
                 "Only insert every Nth depth pixel (in both axes) -- a "
                 "compute-budget knob, not a depth-quality one.");
  double occupancy_rate_hz = 5.0;
  app.add_option("--occupancy-rate-hz", occupancy_rate_hz,
                 "Max rate to integrate depth frames into the occupancy "
                 "grid -- independent of the depth stream's own ~30fps, "
                 "since mapping doesn't need every frame.");
  // min default: see OccupancyMapper's constructor comment -- without
  // this, active-IR stereo on a close, flat, low-texture wall was
  // observed producing high-confidence FALSE matches many meters away
  // that a confidence threshold alone doesn't catch, live-diagnosed on
  // OAK-D Pro W 2026-09-29.
  int occupancy_min_depth_mm = 150;
  app.add_option("--occupancy-min-depth-mm", occupancy_min_depth_mm,
                 "Reject depth pixels closer than this (mm) -- 0 disables.");
  // History, 2026-09-29: 3000 -> pileup right at 3000mm while rotating
  // in a corner. Raised to 6000 assuming an open room -- pileup just
  // moved to 6000mm instead, and the user confirmed the real walls are
  // NOT farther than ~3m (sitting in a corner, rotating in place). So
  // that pileup was never a real room boundary either time -- it's
  // confidently-wrong far matches (see setConfidenceThreshold's comment
  // in oak_d.cpp for the likely cause: grazing-angle viewing, hard to
  // avoid while rotating in a corner). Back down to a value grounded in
  // the ACTUAL known room bound now that we have one, plus a margin for
  // corner-to-far-corner diagonal distance -- not a blind guess this
  // time. Tighten toward 3000 or raise toward the room's real diagonal
  // if this specific value doesn't match the actual space.
  int occupancy_max_depth_mm = 3500;
  app.add_option("--occupancy-max-depth-mm", occupancy_max_depth_mm,
                 "Reject depth pixels farther than this (mm) -- 0 disables. "
                 "Filters out coherent false stereo matches on close, "
                 "low-texture surfaces under active IR, not just noise.");

  // OAK-D Pro W only -- silently a no-op on hardware without an IR
  // projector (see OakDDevice::setIrEmitters()'s comment). Off by default,
  // same reasoning as enable_occupancy_mapping above: opt-in for a feature
  // this specific hardware needs recalibration/live validation before
  // being trusted unconditionally. Live-verified crash-safe in isolation
  // via basalt_test_ir_emitters before being wired in here.
  bool enable_ir_emitters = false;
  app.add_option("--enable-ir-emitters", enable_ir_emitters,
                 "Turn on the OAK-D Pro W's IR laser dot projector + flood "
                 "light for active depth (see basalt_test_ir_emitters).");
  // 0.17, not the previous 0.5 -- setIrLaserDotProjectorIntensity()'s own
  // doc comment says intensity is normalized to up to ~1200mA, and
  // Luxonis's own OAK-D Pro W docs specifically recommend ~200mA
  // (200/1200 ~= 0.17) for improving depth on blank/textureless walls.
  // 0.5 (~600mA) is ~3x that -- plausible over-drive at the ~0.4-0.5m
  // range this rig is tested at, saturating/blooming the dot pattern on a
  // close flat wall into a coherent but WRONG set of high-confidence
  // stereo matches (not just noise -- see OccupancyMapper's min/max depth
  // filter comment for that same failure mode, diagnosed 2026-09-29 as a
  // depth-VALUE problem, not a coordinate/math one). Revisit empirically;
  // this is the documented starting point, not a live-tuned final value.
  double ir_laser_intensity = 0.17;
  app.add_option("--ir-laser-intensity", ir_laser_intensity,
                 "IR laser dot projector intensity, 0.0-1.0. Only applied "
                 "if --enable-ir-emitters is set.");
  double ir_flood_intensity = 0.0;
  app.add_option("--ir-flood-intensity", ir_flood_intensity,
                 "IR flood light intensity, 0.0-1.0. Only applied if "
                 "--enable-ir-emitters is set. Off by default -- the dot "
                 "projector alone is what active depth needs; flood mainly "
                 "helps plain image brightness in the dark.");

  // Default: a fresh timestamped folder per run, so repeated test runs
  // don't clobber each other and can be compared later -- see
  // writeTrajectoryLogs() for what actually gets written into it.
  // Offline reconstruction input (see scripts/offline_tsdf.py): every
  // processed depth frame as a 16-bit PNG plus the full raw VIO
  // trajectory, so poses can be interpolated at each depth timestamp
  // offline instead of trusting the live nearest-pose pairing.
  bool depth_full_res = true;
  app.add_option("--depth-full-res", depth_full_res,
                 "Depth from the mono sensors' native 1280x800 at 10 fps (VIO "
                 "stays 640x480 30 fps). false = old 640x480 crop depth.");
  std::string record_depth_dir;
  app.add_option("--record-depth-dir", record_depth_dir,
                 "Record depth frames + raw trajectory here for offline TSDF "
                 "reconstruction (enables the depth stream).");

  std::string log_dir;
  app.add_option("--log-dir", log_dir,
                 "Directory to save raw/corrected trajectories and a drift "
                 "summary to on exit (default: ./run_logs/<timestamp>/).");

  // Accelerometer bias (especially its component along gravity) is only
  // observable through translational motion -- during a stationary hold,
  // the optimizer converges the bias state to whatever locally-consistent
  // value the very limited available constraints allow, which is not
  // necessarily close to the true physical bias. Starting from a flat
  // Eigen::Vector3d::Zero() guess every run means an unobserved/wrong
  // component just integrates as pure drift for the rest of a stationary
  // test. These defaults are the converged accel/gyro bias observed on a
  // real run of this OAK-D Lite unit (run_logs/20260917_052055,
  // [VIO-BIAS] log) -- a better starting point than zero, not a certified
  // calibration. Override once a proper motion-excited calibration
  // exists, or if this unit's IMU characteristics change.
  std::vector<double> accel_bias_init = {0.0297, -0.0447, 0.0431};
  app.add_option("--accel-bias-init", accel_bias_init,
                 "Initial accelerometer bias guess [x y z], m/s^2 (default: "
                 "this unit's last observed converged value, not a "
                 "certified calibration).")
      ->expected(3);
  std::vector<double> gyro_bias_init = {0.00476, -0.0001, 0.00149};
  app.add_option("--gyro-bias-init", gyro_bias_init,
                 "Initial gyroscope bias guess [x y z], rad/s (default: "
                 "this unit's last observed converged value, not a "
                 "certified calibration).")
      ->expected(3);

  // See VioConfig::optical_flow_stereo_seed_depths_m's comment -- lets a
  // compute-constrained platform (e.g. a Pi5) run fewer stereo-seed depth
  // hypotheses than the laptop-validated default without needing a full
  // --config-path file just for this one value. Empty means "use
  // whatever config_path/the default already set" (CLI11 leaves the
  // vio_config field untouched if this option is never passed).
  std::vector<double> stereo_seed_depths;
  app.add_option("--stereo-seed-depths", stereo_seed_depths,
                 "Override the cam0->cam1 stereo-seed depth hypotheses "
                 "(meters, any count) -- default is the laptop-validated "
                 "7-depth set; pass fewer on compute-constrained hardware.");

  // Same reasoning as --stereo-seed-depths above -- lets this be swept
  // (e.g. testing whether a deeper pyramid tolerates larger inter-frame
  // displacement before losing tracked points, 2026-09-23 investigation)
  // without a full --config-path file per value tried. 0 means "use
  // whatever config_path/the default already set" (CLI11 default), since
  // 0 levels (single-scale KLT) is never a real config to test here.
  int optical_flow_levels = 0;
  app.add_option("--optical-flow-levels", optical_flow_levels,
                 "Override the KLT pyramid depth (default 3) -- more "
                 "levels tolerate larger inter-frame pixel displacement "
                 "per tracked point, at higher per-frame compute cost.");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  if (log_dir.empty()) {
    auto now = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now());
    std::tm tm_buf;
    localtime_r(&now, &tm_buf);
    char buf[64];
    std::strftime(buf, sizeof(buf), "run_logs/%Y%m%d_%H%M%S", &tm_buf);
    log_dir = buf;
  }

  basalt::fs::create_directories(log_dir);

  std::ofstream console_log_file(log_dir + "/console.log");
  std::streambuf* orig_cout_buf = std::cout.rdbuf();
  std::streambuf* orig_cerr_buf = std::cerr.rdbuf();
  std::unique_ptr<TeeStreambuf> tee_cout, tee_cerr;
  if (console_log_file.is_open()) {
    tee_cout.reset(new TeeStreambuf(orig_cout_buf, console_log_file.rdbuf()));
    tee_cerr.reset(new TeeStreambuf(orig_cerr_buf, console_log_file.rdbuf()));
    std::cout.rdbuf(tee_cout.get());
    std::cerr.rdbuf(tee_cerr.get());
  } else {
    std::cerr << "Warning: could not open " << log_dir
              << "/console.log for writing; console output will not be "
                 "saved for this run."
              << std::endl;
  }

  // global thread limit is in effect until global_control object is destroyed
  std::unique_ptr<tbb::global_control> tbb_global_control;
  if (num_threads > 0) {
    tbb_global_control = std::make_unique<tbb::global_control>(
        tbb::global_control::max_allowed_parallelism, num_threads);
  }

  if (!config_path.empty()) {
    vio_config.load(config_path);
  } else {
    vio_config.optical_flow_skip_frames = 2;
  }
  if (!stereo_seed_depths.empty()) {
    vio_config.optical_flow_stereo_seed_depths_m = stereo_seed_depths;
  }
  if (optical_flow_levels > 0) {
    vio_config.optical_flow_levels = optical_flow_levels;
  }

  load_data(cam_calib_path);

  const bool record_depth = !record_depth_dir.empty();
  oakd_device.reset(
      new basalt::OakDDevice(enable_occupancy_mapping || record_depth,
                              depth_full_res));

  try {
    oakd_device->start();
  } catch (const std::exception& e) {
    std::cerr << "Failed to start OAK-D Lite: " << e.what() << std::endl;
    return 1;
  }

  if (enable_ir_emitters) {
    oakd_device->setIrEmitters(static_cast<float>(ir_laser_intensity),
                                static_cast<float>(ir_flood_intensity));
  }

  opt_flow_ptr = basalt::OpticalFlowFactory::getOpticalFlow(vio_config, calib);

  vio = basalt::VioEstimatorFactory::getVioEstimator(
      vio_config, calib, basalt::constants::g, true, use_double);

  // Wire the real IMU queue in BEFORE initialize(): initialize() spawns a
  // background thread that immediately does a blocking pop off
  // vio->imu_data_queue expecting the device to already be pushing into
  // it. This used to run the other way around (device pointed at a null
  // IMU queue, then initialize(), then the real queue wired in after),
  // which reproduced a same-run "first IMU measurment is nullptr" abort
  // on the Pi5 in headless (--show-gui false) testing.
  oakd_device->setOutputQueues(&opt_flow_ptr->input_queue,
                               &vio->imu_data_queue);
  imu_tap_queue.set_capacity(300);
  oakd_device->setImuTapQueue(&imu_tap_queue);
  vio->initialize(
      Eigen::Vector3d(gyro_bias_init[0], gyro_bias_init[1], gyro_bias_init[2]),
      Eigen::Vector3d(accel_bias_init[0], accel_bias_init[1],
                      accel_bias_init[2]));

  opt_flow_ptr->output_queue = &vio->vision_data_queue;
  if (show_gui) vio->out_vis_queue = &out_vis_queue;
  vio->out_state_queue = &out_state_queue;

  basalt::MargDataSaver::Ptr marg_data_saver;
  if (!marg_data_path.empty()) {
    marg_data_saver.reset(new basalt::MargDataSaver(marg_data_path));
  }

  if (online_loop_closure_enabled) {
    online_loop_closure.reset(
        new basalt::OnlineLoopClosure(calib, vio_config));
    online_loop_closure->start();
  }

  // out_marg_queue is a single pointer, but marg_data_saver (feeds the
  // offline basalt_mapper later) and online_loop_closure (live correction
  // now) can both legitimately want the same stream -- fan it out to
  // whichever of the two are actually active instead of forcing a choice
  // between them (see MargDataFanOut's own comment for why sharing the
  // same MargData::Ptr between both is safe).
  if (marg_data_saver && online_loop_closure) {
    marg_fan_out.reset(new basalt::MargDataFanOut(
        {&marg_data_saver->in_marg_queue, &online_loop_closure->input_queue}));
    vio->out_marg_queue = &marg_fan_out->in_queue;
  } else if (marg_data_saver) {
    vio->out_marg_queue = &marg_data_saver->in_marg_queue;
  } else if (online_loop_closure) {
    vio->out_marg_queue = &online_loop_closure->input_queue;
  }

  if (!dashboard_host.empty()) {
    dashboard_client.reset(
        new basalt::DashboardClient(dashboard_host, dashboard_port));
    dashboard_client->start();
  }

  if (enable_occupancy_mapping) {
    // Luxonis's own factory calibration, NOT Basalt's calib -- see
    // OccupancyMapper::DepthIntrinsics's header comment for why using
    // Basalt's own (RAW-lens) camera model here produced a severely
    // warped point cloud on OAK-D Pro W.
    auto oakd_intr = oakd_device->getDepthIntrinsics();
    basalt::DepthIntrinsics depth_intr;
    depth_intr.fx = oakd_intr.fx;
    depth_intr.fy = oakd_intr.fy;
    depth_intr.cx = oakd_intr.cx;
    depth_intr.cy = oakd_intr.cy;
    depth_intr.width = oakd_intr.width;
    depth_intr.height = oakd_intr.height;
    occupancy_mapper.reset(new basalt::OccupancyMapper(
        depth_intr, occupancy_voxel_size, occupancy_depth_stride,
        occupancy_min_depth_mm, occupancy_max_depth_mm));
    occupancy_mapper->start();
    // No setPoseLookup(): rebuild-on-loop-closure is disabled. Live
    // loop-closure corrections measured ~0 (median 0 m, p95 2.8 cm) on
    // real panning runs, so re-posing had nothing to fix, while clear()
    // isn't recorded by octomap change detection -- rebuilds only ever
    // ADDED voxels on the dashboard. The clean map comes from the offline
    // TSDF path (--record-depth-dir) instead.
  }

  std::ofstream rec_frames, rec_traj;
  if (record_depth) {
    basalt::fs::create_directories(record_depth_dir + "/depth");
    auto intr = oakd_device->getDepthIntrinsics();
    const Sophus::SE3d& T_i_c = calib.T_i_c[0];
    Eigen::Quaterniond q = T_i_c.so3().unit_quaternion();
    std::ofstream meta(record_depth_dir + "/meta.json");
    meta.precision(10);
    meta << "{\n  \"fx\": " << intr.fx << ", \"fy\": " << intr.fy
         << ", \"cx\": " << intr.cx << ", \"cy\": " << intr.cy
         << ",\n  \"width\": " << intr.width << ", \"height\": " << intr.height
         << ",\n  \"depth_scale\": 1000.0,\n  \"T_i_c\": {\"px\": "
         << T_i_c.translation().x() << ", \"py\": " << T_i_c.translation().y()
         << ", \"pz\": " << T_i_c.translation().z() << ", \"qx\": " << q.x()
         << ", \"qy\": " << q.y() << ", \"qz\": " << q.z() << ", \"qw\": "
         << q.w() << "}\n}\n";
    rec_frames.open(record_depth_dir + "/frames.csv");
    rec_frames << "t_ns,file" << std::endl;
    rec_traj.open(record_depth_dir + "/trajectory.csv");
    rec_traj << "t_ns,tx,ty,tz,qx,qy,qz,qw" << std::endl;
    rec_traj.precision(10);
    std::cout << "[RECORD] depth + trajectory -> " << record_depth_dir
              << std::endl;
    if (dashboard_client) {
      dashboard_client->setRunInfo(
          basalt::fs::absolute(record_depth_dir).string());
    }
  }

  if (occupancy_mapper || record_depth) {
    depth_queue.set_capacity(4);
    oakd_device->setDepthOutputQueue(&depth_queue);
  }

  vio_data_log.Clear();
  vio_plot_queue.set_capacity(10000);

  std::shared_ptr<std::thread> t3;

  if (show_gui)
    t3.reset(new std::thread([&]() {
      basalt::VioVisualizationData::Ptr data;
      while (!terminate) {
        out_vis_queue.pop(data);

        if (!data.get()) break;

        std::lock_guard<std::mutex> lock(curr_vis_data_mutex);
        curr_vis_data = data;
      }

      std::cout << "Finished t3" << std::endl;
    }));

  std::thread t4([&]() {
    basalt::PoseVelBiasState<double>::Ptr data;

    while (!terminate) {
      out_state_queue.pop(data);

      if (!data.get()) break;

      int64_t t_ns = data->t_ns;

      {
        std::lock_guard<std::mutex> lock(vio_state_mutex);
        if (curr_t_ns < 0) curr_t_ns = t_ns;
      }

      Sophus::SE3d T_w_i = data->T_w_i;
      Eigen::Vector3d vel_w_i = data->vel_w_i;
      Eigen::Vector3d bg = data->bias_gyro;
      Eigen::Vector3d ba = data->bias_accel;

      {
        std::lock_guard<std::mutex> lock(vio_state_mutex);
        vio_t_ns.emplace_back(data->t_ns);
        vio_t_w_i.emplace_back(T_w_i.translation());
        curr_raw_pose = T_w_i;
        raw_pose_history.emplace_back(data->t_ns, T_w_i);
        while (raw_pose_history.size() > kRawPoseHistoryCap) {
          raw_pose_history.pop_front();
        }
      }

      if (rec_traj.is_open()) {
        Eigen::Quaterniond q = T_w_i.so3().unit_quaternion();
        const Eigen::Vector3d& p = T_w_i.translation();
        rec_traj << t_ns << ',' << p.x() << ',' << p.y() << ',' << p.z() << ','
                 << q.x() << ',' << q.y() << ',' << q.z() << ',' << q.w()
                 << std::endl;  // flush: the process can abort on exit, losing buffered rows
      }

      if (dashboard_client) {
        Eigen::Quaterniond q = T_w_i.so3().unit_quaternion();
        dashboard_client->sendPose(
            t_ns, /*corrected=*/false, T_w_i.translation(),
            Eigen::Vector4d(q.x(), q.y(), q.z(), q.w()), &vel_w_i);

        if (online_loop_closure) {
          Sophus::SE3d T_corrected;
          // Smoothed (rebased-by-raw-motion), not the newest pose-graph
          // node's own position directly -- see getSmoothedCorrectedPose()
          // comment. The latter jumped visibly on every re-solve, most
          // noticeably with the camera near-stationary (confirmed live:
          // 197 closures over 325 keyframes in one such test).
          if (online_loop_closure->getSmoothedCorrectedPose(T_w_i, T_corrected)) {
            Eigen::Quaterniond qc = T_corrected.so3().unit_quaternion();
            dashboard_client->sendPose(
                t_ns, /*corrected=*/true, T_corrected.translation(),
                Eigen::Vector4d(qc.x(), qc.y(), qc.z(), qc.w()));
          }
        }
      }

      // Confidence/health signal (see basalt/utils/vio_health.h) --
      // published from this thread specifically because it runs
      // unconditionally on the live drone path, unlike out_vis_queue
      // (only wired when show_gui is true, dead on a real headless
      // flight). Logged locally regardless of whether a dashboard is
      // connected, so this also works fully offline on the bench.
      {
        // Raw counts and ratio, computed up front: tracked_count feeds
        // computeVioConfidence() below, tracked_ratio is kept only for
        // reportTrackingHealth()/logging/dashboard (see
        // VioConfidenceInputs::tracked_count's comment for why the
        // confidence signal itself moved off the ratio).
        int tracked_count = vio->getLatestTrackedCount();
        int total_observed_count = vio->getLatestTotalObservedCount();
        double tracked_ratio = vio->getLatestTrackedRatio();

        basalt::VioConfidenceInputs health_in;
        health_in.tracked_count = tracked_count;
        health_in.numerically_degraded = vio->isDegraded();
        health_in.gyro_norm = vio->getLatestGyroNorm();
        health_in.imu_vision_disagreement = vio->isImuVisionDisagreement();
        if (online_loop_closure) {
          health_in.triangulated_points =
              online_loop_closure->getLatestTriangulatedPoints();
          health_in.recently_forced_drift_release =
              online_loop_closure->isRecentlyForceReleased();
        }

        basalt::VioConfidence health = basalt::computeVioConfidence(health_in);

        // Feeds the drift gate's starvation trigger (see
        // kStarvationMinTrackedCount in online_loop_closure.cpp) --
        // lets it hold the live pose when tracking is starved badly
        // enough that few or no new keyframes are being created, a
        // window checkDriftGate()'s own residual check can otherwise
        // miss entirely.
        if (online_loop_closure) {
          online_loop_closure->reportTrackingHealth(
              tracked_ratio, tracked_count, total_observed_count);
          // See OnlineLoopClosure::reportAccelStability()'s comment --
          // lets a starvation hold's force-release tell a real
          // camera-cover-while-still episode from an ordinary
          // tracking-loss-while-moving one.
          online_loop_closure->reportAccelStability(vio->isLikelyStationary(),
                                                    vio->getLatestAccelStd());
        }

        if (health.primary_reason != "nominal") {
          std::cout << "[VIO-HEALTH] t_ns=" << t_ns
                    << " confidence=" << health.score
                    << " reason=" << health.primary_reason
                    << " tracked_ratio=" << tracked_ratio
                    << " tracked_count=" << tracked_count
                    << " total_observed_count=" << total_observed_count
                    << " triangulated_points="
                    << (health_in.triangulated_points
                            ? std::to_string(*health_in.triangulated_points)
                            : "n/a")
                    << " gyro_norm=" << health_in.gyro_norm
                    << " accel_norm=" << vio->getLatestAccelNorm()
                    << " imu_vision_disagreement_m="
                    << vio->getLatestImuVisionDisagreementM()
                    << " imu_vision_reweight_active="
                    << vio->isImuVisionReweightActive()
                    << " bias_freeze_active=" << vio->isBiasFreezeActive()
                    << std::endl;
        }

        // Bias-state telemetry: printed unconditionally (health-gated
        // above) and throttled to ~1Hz (this block runs at frame rate) so
        // a GOOD run's bias trajectory is visible too, not just a
        // degraded one -- diagnosing a runaway raw trajectory needs to
        // compare both, e.g. two same-config runs where one stayed at
        // 2m drift and one blew up to 300m despite near-identical
        // starting conditions.
        static int bias_log_counter = 0;
        if (++bias_log_counter >= 15) {
          bias_log_counter = 0;
          Eigen::Vector3d accel_bias = vio->getLatestAccelBias();
          Eigen::Vector3d gyro_bias = vio->getLatestGyroBias();
          std::cout << "[VIO-BIAS] t_ns=" << t_ns
                    << " accel_bias=[" << accel_bias.transpose() << "]"
                    << " gyro_bias=[" << gyro_bias.transpose() << "]"
                    << std::endl;
        }

        if (dashboard_client) {
          dashboard_client->sendHealth(
              t_ns, health.score, health.primary_reason,
              health_in.numerically_degraded, health_in.gyro_norm,
              tracked_ratio, health_in.triangulated_points,
              tracked_count, total_observed_count);
        }
      }

      if (show_gui) {
        std::vector<float> vals;
        {
          std::lock_guard<std::mutex> lock(vio_state_mutex);
          vals.push_back((t_ns - curr_t_ns) * 1e-9);
        }

        for (int i = 0; i < 3; i++) vals.push_back(vel_w_i[i]);
        for (int i = 0; i < 3; i++) vals.push_back(T_w_i.translation()[i]);
        for (int i = 0; i < 3; i++) vals.push_back(bg[i]);
        for (int i = 0; i < 3; i++) vals.push_back(ba[i]);

        vio_plot_queue.try_push(vals);
      }
    }

    std::cout << "Finished t4" << std::endl;
  });

  std::shared_ptr<std::thread> t5;

  if (print_queue) {
    t5.reset(new std::thread([&]() {
      while (!terminate) {
        std::cout << "opt_flow_ptr->input_queue "
                  << opt_flow_ptr->input_queue.size()
                  << " opt_flow_ptr->output_queue "
                  << opt_flow_ptr->output_queue->size() << " out_state_queue "
                  << out_state_queue.size() << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
    }));
  }

  // Dashboard map-event/save-map polling. Deliberately its own thread
  // rather than folded into the GUI frame loop below (like
  // drain_vio_plot_queue()/drain_localization_queue() are) -- that loop
  // only runs when show_gui is true, but a real flight (no display
  // attached) needs this working regardless.
  std::thread t6([&]() {
    int last_num_closures = 0;
    while (!terminate) {
      if (online_loop_closure) {
        int n = online_loop_closure->numLoopClosures();
        if (n > last_num_closures) {
          last_num_closures = n;
          if (dashboard_client) {
            // NOT curr_t_ns -- that's only ever set once, to the very
            // first pose (see t4: "if (curr_t_ns < 0) curr_t_ns = t_ns;"),
            // so every loop_closure event was previously stamped with the
            // session's start time regardless of when the closure
            // actually happened (harmless for the real-time dashboard
            // toast, which still popped up at roughly the right moment,
            // but useless for any after-the-fact analysis of the stored
            // JSONL). vio_t_ns.back() is the actual latest timestamp VIO
            // has processed, updated every t4 iteration under the same
            // mutex.
            int64_t t_ns;
            {
              std::lock_guard<std::mutex> lock(vio_state_mutex);
              t_ns = vio_t_ns.empty() ? curr_t_ns : vio_t_ns.back();
            }
            dashboard_client->sendMapEvent(t_ns, "loop_closure");
          }
        }
      }
      if (dashboard_client) {
        if (online_loop_closure) {
          // Drains OnlineLoopClosure::drift_gate_events instead of
          // polling isDriftHeld() -- a real live test found polling
          // (even at this loop's own ~200ms cadence) can miss rapid
          // trip/release cycles entirely, undercounting a genuinely
          // flapping gate down to what looked like one long, unexplained
          // freeze. Draining the queue reports every transition that
          // actually happened, in order, regardless of how fast they
          // cycled between two checks of this loop.
          basalt::DriftGateEvent gate_event = basalt::DriftGateEvent::kTripped;
          while (online_loop_closure->drift_gate_events.try_pop(gate_event)) {
            // Same curr_t_ns bug as the loop_closure event above -- use
            // the actual latest processed timestamp instead.
            int64_t t_ns;
            {
              std::lock_guard<std::mutex> lock(vio_state_mutex);
              t_ns = vio_t_ns.empty() ? curr_t_ns : vio_t_ns.back();
            }
            // MapEventType (schema.py) only has "drift_hold"/"drift_recovered"
            // -- the confirmed-vs-forced distinction (see DriftGateEvent's
            // comment) rides in `detail` instead of a new event string, so
            // it doesn't require a dashboard-side enum change to consume.
            switch (gate_event) {
              case basalt::DriftGateEvent::kTripped:
                dashboard_client->sendMapEvent(t_ns, "drift_hold");
                break;
              case basalt::DriftGateEvent::kReleasedConfirmed:
                dashboard_client->sendMapEvent(t_ns, "drift_recovered",
                                               R"({"forced": false})");
                break;
              case basalt::DriftGateEvent::kReleasedForced:
                dashboard_client->sendMapEvent(t_ns, "drift_recovered",
                                               R"({"forced": true})");
                break;
            }
          }
        }

        std::string cmd_run_id;
        if (dashboard_client->pollSaveMapCommand(cmd_run_id) &&
            online_loop_closure) {
          auto points = online_loop_closure->buildPointCloud();
          std::string ply_path = log_dir + "/live_map.ply";
          if (basalt::writePointCloudPly(ply_path, points)) {
            dashboard_client->sendMapFile("flight map", ply_path);
          }
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    std::cout << "Finished t6" << std::endl;
  });

  // Feeds real depth frames + the current pose into occupancy_mapper.
  // Independent of show_gui for the same reason t6 is -- a real headless
  // flight still needs mapping to run.
  std::shared_ptr<std::thread> t7;
  if (occupancy_mapper || record_depth) {
    t7.reset(new std::thread([&]() {
      basalt::lowerCurrentThreadPriority();  // depth saving / map insertion
      auto last_processed = std::chrono::steady_clock::now();
      const auto min_interval = std::chrono::duration<double>(
          1.0 / std::max(0.1, occupancy_rate_hz));

      int64_t total_added = 0, total_removed = 0;

      while (!terminate) {
        std::shared_ptr<dai::ImgFrame> depth_frame;
        depth_queue.pop(depth_frame);
        if (!depth_frame) break;  // shutdown sentinel from OakDDevice::stop()

        auto now = std::chrono::steady_clock::now();
        if (now - last_processed < min_interval) continue;  // rate budget
        last_processed = now;

        double t_sec = std::chrono::duration<double>(
                           depth_frame->getTimestamp().time_since_epoch())
                           .count();
        int64_t depth_t_ns = static_cast<int64_t>(t_sec * 1e9);

        if (record_depth) {
          std::string name = std::to_string(depth_t_ns) + ".png";
          if (cv::imwrite(record_depth_dir + "/depth/" + name,
                          depth_frame->getCvFrame())) {
            rec_frames << depth_t_ns << ',' << name << std::endl;
          }
        }
        if (!occupancy_mapper) continue;

        // Pose AT THE DEPTH FRAME'S OWN CAPTURE TIME. VIO usually lags the
        // depth stream, so wait (briefly) for a VIO state at or after
        // depth_t_ns instead of silently clamping to an older pose, and
        // skip the frame if the nearest state is still too far away.
        constexpr int64_t kMaxPoseGapNs = 40'000'000;  // 40 ms
        Sophus::SE3d raw_now;
        bool have_pose = false;
        for (int attempt = 0; attempt < 15 && !have_pose && !terminate;
             ++attempt) {
          {
            std::lock_guard<std::mutex> lock(vio_state_mutex);
            if (!raw_pose_history.empty() &&
                raw_pose_history.back().first >= depth_t_ns) {
              auto it = std::lower_bound(
                  raw_pose_history.begin(), raw_pose_history.end(),
                  depth_t_ns,
                  [](const std::pair<int64_t, Sophus::SE3d>& s, int64_t t) {
                    return s.first < t;
                  });
              if (it != raw_pose_history.begin()) {
                auto prev = std::prev(it);
                if (depth_t_ns - prev->first < it->first - depth_t_ns) {
                  it = prev;
                }
              }
              if (std::abs(it->first - depth_t_ns) <= kMaxPoseGapNs) {
                raw_now = it->second;
                have_pose = true;
              }
              break;
            }
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (!have_pose) continue;

        // The RAW VIO pose, not getSmoothedCorrectedPose(): that one is a
        // display pose (frozen during drift-gate holds, interpolated during
        // blends), and inserting depth with it while the camera moves puts
        // copies of surfaces in the wrong place.
        const Sophus::SE3d& pose = raw_now;

        auto input = std::make_shared<basalt::DepthFrameInput>();
        input->t_ns = depth_t_ns;
        input->T_w_c = pose * calib.T_i_c[0];
        // Uncorrected body pose at capture -- retained by OccupancyMapper
        // so a later rebuild() can ask OnlineLoopClosure for a FRESH
        // corrected pose at this same t_ns once the pose graph has moved
        // on, instead of being stuck with whatever correction (if any)
        // was live right now. See DepthFrameInput::T_w_i_raw's comment.
        input->T_w_i_raw = raw_now;
        input->depth_mm = depth_frame->getCvFrame();
        occupancy_mapper->addDepthFrame(input);

        basalt::VoxelDelta delta;
        while (occupancy_mapper->pollVoxelDelta(delta)) {
          total_added += delta.added.size();
          total_removed += delta.removed.size();
          if (dashboard_client) {
            dashboard_client->sendVoxelBatch(input->t_ns, occupancy_voxel_size,
                                              delta.added, delta.removed);
          }
        }
      }

      std::cout << "Finished t7 -- occupancy grid: " << total_added
                << " voxels added, " << total_removed << " removed (net "
                << (total_added - total_removed) << " occupied)"
                << std::endl;
    }));
  }

  // Forwards the raw IMU tap to opt_flow_ptr's rotation-compensated KLT
  // seeding (see FrameToFrameOpticalFlow::addIMUToQueue()'s comment).
  // Independent of show_gui/online_loop_closure for the same reason
  // t6/t7 are -- this needs to run for a real headless flight too.
  std::thread t8([&]() {
    basalt::ImuData<double>::Ptr data;
    while (!terminate) {
      imu_tap_queue.pop(data);
      if (!data) break;  // shutdown sentinel, pushed below
      opt_flow_ptr->addIMUToQueue(data);
    }
    std::cout << "Finished t8" << std::endl;
  });

  if (show_gui) {
    pangolin::CreateWindowAndBind("OAK-D Lite Vio", 1800, 1000);

    glEnable(GL_DEPTH_TEST);

    pangolin::View& img_view_display =
        pangolin::CreateDisplay()
            .SetBounds(0.4, 1.0, pangolin::Attach::Pix(UI_WIDTH), 0.4)
            .SetLayout(pangolin::LayoutEqual);

    pangolin::View& plot_display = pangolin::CreateDisplay().SetBounds(
        0.0, 0.4, pangolin::Attach::Pix(UI_WIDTH), 1.0);

    plotter =
        new pangolin::Plotter(&imu_data_log, 0.0, 100, -3.0, 3.0, 0.01f, 0.01f);
    plot_display.AddDisplay(*plotter);

    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0,
                                          pangolin::Attach::Pix(UI_WIDTH));

    std::vector<std::shared_ptr<pangolin::ImageView>> img_view;
    while (img_view.size() < calib.intrinsics.size()) {
      std::shared_ptr<pangolin::ImageView> iv(new pangolin::ImageView);

      size_t idx = img_view.size();
      img_view.push_back(iv);

      img_view_display.AddDisplay(*iv);
      iv->extern_draw_function =
          std::bind(&draw_image_overlay, std::placeholders::_1, idx);
    }

    Eigen::Vector3d cam_p(0.5, -2, -2);
    cam_p = vio->getT_w_i_init().so3() * calib.T_i_c[0].so3() * cam_p;
    cam_p[2] = 1;

    pangolin::OpenGlRenderState camera(
        pangolin::ProjectionMatrix(640, 480, 400, 400, 320, 240, 0.001, 10000),
        pangolin::ModelViewLookAt(cam_p[0], cam_p[1], cam_p[2], 0, 0, 0,
                                  pangolin::AxisZ));

    pangolin::View& display3D =
        pangolin::CreateDisplay()
            .SetAspect(-640 / 480.0)
            .SetBounds(0.4, 1.0, 0.4, 1.0)
            .SetHandler(new pangolin::Handler3D(camera));

    while (!pangolin::ShouldQuit()) {
      drain_vio_plot_queue();
      drain_localization_queue();

      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      if (follow) {
        // Follow the loop-closure-corrected pose (blue line) when
        // available, not the raw VIO pose (red line) -- the raw pose can
        // sit meters away from the corrected one right after a big loop
        // closure snap, which used to make the camera chase the wrong
        // trajectory. Falls back to the raw pose only when loop closure
        // isn't running at all (or hasn't produced a keyframe yet).
        // Smoothed, not the newest node's raw position, for the same
        // reason as t4/t7 above -- see getSmoothedCorrectedPose().
        Sophus::SE3d T_w_i;
        bool have_pose = false;

        auto vis_data = get_curr_vis_data_snapshot();
        if (vis_data.get() && !vis_data->states.empty()) {
          Sophus::SE3d raw_now = vis_data->states.back();
          if (!(online_loop_closure &&
                online_loop_closure->getSmoothedCorrectedPose(raw_now, T_w_i))) {
            T_w_i = raw_now;
          }
          have_pose = true;
        }

        if (have_pose) {
          T_w_i.so3() = Sophus::SO3d();
          camera.Follow(T_w_i.matrix());
        }
      }

      display3D.Activate(camera);
      glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

      draw_scene();

      img_view_display.Activate();

      {
        pangolin::GlPixFormat fmt;
        fmt.glformat = GL_LUMINANCE;
        fmt.gltype = GL_UNSIGNED_SHORT;
        fmt.scalable_internal_format = GL_LUMINANCE16;

        auto vis_data = get_curr_vis_data_snapshot();
        if (vis_data.get() && vis_data->opt_flow_res.get() &&
            vis_data->opt_flow_res->input_images.get()) {
          auto& img_data = vis_data->opt_flow_res->input_images->img_data;

          for (size_t cam_id = 0; cam_id < basalt::OakDDevice::NUM_CAMS;
               cam_id++) {
            if (img_data[cam_id].img.get())
              img_view[cam_id]->SetImage(
                  img_data[cam_id].img->ptr, img_data[cam_id].img->w,
                  img_data[cam_id].img->h, img_data[cam_id].img->pitch, fmt);
          }
        }

        draw_plots();
      }

      if (show_est_vel.GuiChanged() || show_est_pos.GuiChanged() ||
          show_est_ba.GuiChanged() || show_est_bg.GuiChanged()) {
        draw_plots();
      }

      pangolin::FinishFrame();
    }
  } else {
    // No GUI event loop to keep the process alive while flying headless --
    // block here until Ctrl-C/SIGTERM (handle_shutdown_signal) requests a
    // clean shutdown, the same role pangolin::ShouldQuit() plays above.
    // Without this, main() previously fell straight through to shutdown
    // within milliseconds of starting whenever --show-gui was false --
    // headless flight never actually ran at all before this fix.
    while (!terminate) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }

  oakd_device->stop();
  vio->maybe_join();
  terminate = true;

  if (online_loop_closure) online_loop_closure->stop();
  if (occupancy_mapper) occupancy_mapper->stop();

  // Push nullptr to output queues to unblock waiting threads. imu_tap_queue
  // needs its own push -- OakDDevice::stop() only sentinels the queues it
  // knows about from setOutputQueues()/setDepthOutputQueue(), not this one
  // (see setImuTapQueue()'s comment: a raw, independent tap).
  out_vis_queue.push(nullptr);
  out_state_queue.push(nullptr);
  imu_tap_queue.push(nullptr);

  if (t3.get()) t3->join();
  t4.join();
  if (t5.get()) t5->join();
  t6.join();
  if (t7.get()) t7->join();
  t8.join();

  if (dashboard_client) dashboard_client->stop();

  write_trajectory_logs(log_dir);

  std::cout.rdbuf(orig_cout_buf);
  std::cerr.rdbuf(orig_cerr_buf);

  return 0;
}

void draw_image_overlay(pangolin::View& v, size_t cam_id) {
  UNUSED(v);
  auto vis_data = get_curr_vis_data_snapshot();

  if (show_obs) {
    glLineWidth(1.0);
    glColor3f(1.0, 0.0, 0.0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (vis_data.get() && cam_id < vis_data->projections.size()) {
      const auto& points = vis_data->projections[cam_id];

      if (!points.empty()) {
        double min_id = points[0][2], max_id = points[0][2];

        for (const auto& points2 : vis_data->projections)
          for (const auto& p : points2) {
            min_id = std::min(min_id, p[2]);
            max_id = std::max(max_id, p[2]);
          }

        for (const auto& c : points) {
          const float radius = 6.5;

          float r, g, b;
          getcolor(c[2] - min_id, max_id - min_id, b, g, r);
          glColor3f(r, g, b);

          pangolin::glDrawCirclePerimeter(c[0], c[1], radius);

          if (show_ids)
            pangolin::default_font().Text("%d", int(c[3])).Draw(c[0], c[1]);
        }
      }

      glColor3f(1.0, 0.0, 0.0);
      pangolin::default_font()
          .Text("Tracked %d points", points.size())
          .Draw(5, 20);
    }
  }

  // Live position/orientation HUD, drawn once (on cam 0's view) regardless
  // of show_obs, using the latest estimated body pose T_w_i.
  if (cam_id == 0 && vis_data.get() && !vis_data->states.empty()) {
    Sophus::SE3d T_w_i = vis_data->states.back();
    Eigen::Vector3d p = T_w_i.translation();
    Eigen::Vector3d rpy_deg = T_w_i.so3().unit_quaternion().toRotationMatrix().eulerAngles(0, 1, 2) *
                              (180.0 / M_PI);

    glColor3f(0.0, 1.0, 0.0);
    pangolin::default_font()
        .Text("pos (m):   x % .3f  y % .3f  z % .3f", p.x(), p.y(), p.z())
        .Draw(5, 460);
    pangolin::default_font()
        .Text("rpy (deg): r % .1f  p % .1f  y % .1f", rpy_deg.x(), rpy_deg.y(),
              rpy_deg.z())
        .Draw(5, 440);

    if (online_loop_closure) {
      Sophus::SE3d T_corrected;
      if (online_loop_closure->getSmoothedCorrectedPose(T_w_i, T_corrected)) {
        Eigen::Vector3d pc = T_corrected.translation();
        glColor3f(0.0, 0.5, 1.0);
        pangolin::default_font()
            .Text("corrected: x % .3f  y % .3f  z % .3f (loops: %d)", pc.x(),
                  pc.y(), pc.z(), online_loop_closure->numLoopClosures())
            .Draw(5, 420);
      }

      // Latest single verified match, shown independently of the
      // pose-graph-corrected line above -- lets a quick "look at a place
      // you've been before" test show a result immediately, without doing
      // a full walk-and-return and reading drift off the trajectory log.
      if (has_localization) {
        Eigen::Vector3d rel_t =
            latest_localization.T_reference_current.translation();
        int64_t t0;
        {
          std::lock_guard<std::mutex> lock(vio_state_mutex);
          t0 = curr_t_ns;
        }
        double ref_age_s =
            t0 >= 0 ? (latest_localization.reference_t_ns - t0) * 1e-9 : 0.0;
        glColor3f(1.0, 0.5, 0.0);
        pangolin::default_font()
            .Text(
                "localize: ref@%.1fs  rel_t=[% .3f % .3f % .3f]  inliers=%d",
                ref_age_s, rel_t.x(), rel_t.y(), rel_t.z(),
                latest_localization.num_inliers)
            .Draw(5, 400);
      }
    }
  }
}

void draw_scene() {
  glPointSize(3);
  glColor3f(1.0, 0.0, 0.0);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  if (show_raw_traj) {
    glColor3ubv(cam_color);
    Eigen::aligned_vector<Eigen::Vector3d> sub_gt;
    {
      std::lock_guard<std::mutex> lock(vio_state_mutex);
      sub_gt = vio_t_w_i;
    }
    pangolin::glDrawLineStrip(sub_gt);
  }

  // Loop-closure-corrected trajectory, drawn as a second, distinctly
  // colored line alongside the raw (uncorrected) one above -- before any
  // loop closure fires this exactly overlays the raw line, since each new
  // node is seeded from the raw pose.
  if (online_loop_closure) {
    glColor3f(0.0, 0.5, 1.0);  // blue, distinct from cam_color
    glLineWidth(2.0);
    pangolin::glDrawLineStrip(online_loop_closure->getCorrectedTrajectory());
    glLineWidth(1.0);
  }

  auto vis_data = get_curr_vis_data_snapshot();
  if (vis_data.get()) {
    if (show_raw_traj) {
      for (const auto& p : vis_data->states)
        for (const auto& t_i_c : calib.T_i_c)
          render_camera((p * t_i_c).matrix(), 2.0f, state_color, 0.1f);

      for (const auto& p : vis_data->frames)
        for (const auto& t_i_c : calib.T_i_c)
          render_camera((p * t_i_c).matrix(), 2.0f, pose_color, 0.1f);

      for (const auto& t_i_c : calib.T_i_c)
        render_camera((vis_data->states.back() * t_i_c).matrix(), 2.0f,
                      cam_color, 0.1f);
    }

    // "You are here" marker at the CORRECTED pose, matching the blue line
    // and the HUD's corrected:/localize: numbers -- without this, the only
    // drawn camera icon was always the raw pose (cam_color, above), which
    // can sit meters away from the corrected pose right after a big loop
    // closure snap and looks like the marker "teleported" relative to what
    // the screen otherwise reports.
    if (online_loop_closure && vis_data.get() && !vis_data->states.empty()) {
      static const uint8_t corrected_cam_color[3]{0, 128, 255};
      Sophus::SE3d T_w_i_corrected;
      if (online_loop_closure->getSmoothedCorrectedPose(vis_data->states.back(),
                                                          T_w_i_corrected)) {
        for (const auto& t_i_c : calib.T_i_c)
          render_camera((T_w_i_corrected * t_i_c).matrix(), 2.0f,
                        corrected_cam_color, 0.1f);
      }
    }

    glColor3ubv(pose_color);
    pangolin::glDrawPoints(vis_data->points);
  }

  pangolin::glDrawAxis(Sophus::SE3d().matrix(), 1.0);
}

void load_data(const std::string& calib_path) {
  std::ifstream os(calib_path, std::ios::binary);

  if (os.is_open()) {
    cereal::JSONInputArchive archive(os);
    archive(calib);
    std::cout << "Loaded camera with " << calib.intrinsics.size() << " cameras"
              << std::endl;

  } else {
    std::cerr << "could not load camera calibration " << calib_path
              << std::endl;
    std::abort();
  }
}

void write_trajectory_logs(const std::string& log_dir) {
  basalt::fs::create_directories(log_dir);

  std::vector<int64_t> raw_t_ns;
  Eigen::aligned_vector<Eigen::Vector3d> raw_pos;
  {
    std::lock_guard<std::mutex> lock(vio_state_mutex);
    raw_t_ns = vio_t_ns;
    raw_pos = vio_t_w_i;
  }

  {
    std::ofstream os(log_dir + "/raw_trajectory.txt");
    os << "# t_ns x y z\n";
    for (size_t i = 0; i < raw_pos.size(); i++) {
      os << raw_t_ns[i] << " " << raw_pos[i].x() << " " << raw_pos[i].y()
         << " " << raw_pos[i].z() << "\n";
    }
  }

  double raw_drift = -1;
  if (raw_pos.size() >= 2) raw_drift = (raw_pos.back() - raw_pos.front()).norm();

  double corrected_drift = -1;
  size_t num_corrected = 0;
  int num_closures = 0;
  if (online_loop_closure) {
    std::vector<int64_t> corr_t_ns;
    Eigen::aligned_vector<Eigen::Vector3d> corr_pos;
    online_loop_closure->getCorrectedTrajectoryWithTimestamps(corr_t_ns,
                                                               corr_pos);

    std::ofstream os(log_dir + "/corrected_trajectory.txt");
    os << "# t_ns x y z\n";
    for (size_t i = 0; i < corr_pos.size(); i++) {
      os << corr_t_ns[i] << " " << corr_pos[i].x() << " " << corr_pos[i].y()
         << " " << corr_pos[i].z() << "\n";
    }

    num_corrected = corr_pos.size();
    if (corr_pos.size() >= 2)
      corrected_drift = (corr_pos.back() - corr_pos.front()).norm();
    num_closures = online_loop_closure->numLoopClosures();
  }

  {
    std::ofstream os(log_dir + "/summary.txt");
    os << "raw_trajectory_points: " << raw_pos.size() << "\n";
    os << "raw_start_to_end_distance_m: " << raw_drift << "\n";
    if (online_loop_closure) {
      os << "corrected_trajectory_points: " << num_corrected << "\n";
      os << "corrected_start_to_end_distance_m: " << corrected_drift << "\n";
      os << "num_loop_closures: " << num_closures << "\n";
    }
  }

  std::cout << "[LOG] Saved trajectory logs to " << log_dir << "\n"
            << "      raw start-to-end distance: " << raw_drift << " m";
  if (online_loop_closure) {
    std::cout << " | corrected: " << corrected_drift << " m (" << num_closures
              << " closures)";
  }
  std::cout << std::endl;
}

basalt::VioVisualizationData::Ptr get_curr_vis_data_snapshot() {
  std::lock_guard<std::mutex> lock(curr_vis_data_mutex);
  return curr_vis_data;
}

void draw_plots() {
  plotter->ClearSeries();
  plotter->ClearMarkers();

  if (show_est_pos) {
    plotter->AddSeries("$0", "$4", pangolin::DrawingModeLine,
                       pangolin::Colour::Red(), "position x", &vio_data_log);
    plotter->AddSeries("$0", "$5", pangolin::DrawingModeLine,
                       pangolin::Colour::Green(), "position y", &vio_data_log);
    plotter->AddSeries("$0", "$6", pangolin::DrawingModeLine,
                       pangolin::Colour::Blue(), "position z", &vio_data_log);
  }

  if (show_est_vel) {
    plotter->AddSeries("$0", "$1", pangolin::DrawingModeLine,
                       pangolin::Colour::Red(), "velocity x", &vio_data_log);
    plotter->AddSeries("$0", "$2", pangolin::DrawingModeLine,
                       pangolin::Colour::Green(), "velocity y", &vio_data_log);
    plotter->AddSeries("$0", "$3", pangolin::DrawingModeLine,
                       pangolin::Colour::Blue(), "velocity z", &vio_data_log);
  }

  if (show_est_bg) {
    plotter->AddSeries("$0", "$7", pangolin::DrawingModeLine,
                       pangolin::Colour::Red(), "gyro bias x", &vio_data_log);
    plotter->AddSeries("$0", "$8", pangolin::DrawingModeLine,
                       pangolin::Colour::Green(), "gyro bias y", &vio_data_log);
    plotter->AddSeries("$0", "$9", pangolin::DrawingModeLine,
                       pangolin::Colour::Blue(), "gyro bias z", &vio_data_log);
  }

  if (show_est_ba) {
    plotter->AddSeries("$0", "$10", pangolin::DrawingModeLine,
                       pangolin::Colour::Red(), "accel bias x", &vio_data_log);
    plotter->AddSeries("$0", "$11", pangolin::DrawingModeLine,
                       pangolin::Colour::Green(), "accel bias y",
                       &vio_data_log);
    plotter->AddSeries("$0", "$12", pangolin::DrawingModeLine,
                       pangolin::Colour::Blue(), "accel bias z", &vio_data_log);
  }

  auto last_img_data = oakd_device->getLastImageData();
  if (last_img_data.get()) {
    double t = last_img_data->t_ns * 1e-9;
    plotter->AddMarker(pangolin::Marker::Vertical, t, pangolin::Marker::Equal,
                       pangolin::Colour::White());
  }
}

void drain_vio_plot_queue() {
  std::vector<float> vals;
  while (vio_plot_queue.try_pop(vals)) {
    vio_data_log.Log(vals);
  }
}

void drain_localization_queue() {
  if (!online_loop_closure) return;
  basalt::LocalizationResult loc;
  while (online_loop_closure->localization_queue.try_pop(loc)) {
    latest_localization = loc;
    has_localization = true;
  }
}
