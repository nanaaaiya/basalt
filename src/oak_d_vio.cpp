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
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

#include <sophus/se3.hpp>

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
#include <basalt/io/dataset_io.h>
#include <basalt/io/marg_data_io.h>
#include <basalt/utils/filesystem.h>
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
class TeeStreambuf : public std::streambuf {
 public:
  TeeStreambuf(std::streambuf* a, std::streambuf* b) : a_(a), b_(b) {}

 protected:
  int overflow(int c) override {
    if (c == EOF) return !EOF;
    bool ok_a = a_->sputc(static_cast<char>(c)) != EOF;
    bool ok_b = b_->sputc(static_cast<char>(c)) != EOF;
    return (ok_a && ok_b) ? c : EOF;
  }

  int sync() override {
    int ra = a_->pubsync();
    int rb = b_->pubsync();
    return (ra == 0 && rb == 0) ? 0 : -1;
  }

 private:
  std::streambuf* a_;
  std::streambuf* b_;
};

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
std::mutex vio_state_mutex;
tbb::concurrent_bounded_queue<std::vector<float>> vio_plot_queue;

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

  // Default: a fresh timestamped folder per run, so repeated test runs
  // don't clobber each other and can be compared later -- see
  // writeTrajectoryLogs() for what actually gets written into it.
  std::string log_dir;
  app.add_option("--log-dir", log_dir,
                 "Directory to save raw/corrected trajectories and a drift "
                 "summary to on exit (default: ./run_logs/<timestamp>/).");

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

  load_data(cam_calib_path);

  oakd_device.reset(new basalt::OakDDevice);

  try {
    oakd_device->start();
  } catch (const std::exception& e) {
    std::cerr << "Failed to start OAK-D Lite: " << e.what() << std::endl;
    return 1;
  }

  opt_flow_ptr = basalt::OpticalFlowFactory::getOpticalFlow(vio_config, calib);
  oakd_device->setOutputQueues(&opt_flow_ptr->input_queue, nullptr);

  vio = basalt::VioEstimatorFactory::getVioEstimator(
      vio_config, calib, basalt::constants::g, true, use_double);
  vio->initialize(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  oakd_device->setOutputQueues(&opt_flow_ptr->input_queue,
                               &vio->imu_data_queue);

  opt_flow_ptr->output_queue = &vio->vision_data_queue;
  if (show_gui) vio->out_vis_queue = &out_vis_queue;
  vio->out_state_queue = &out_state_queue;

  basalt::MargDataSaver::Ptr marg_data_saver;

  if (!marg_data_path.empty()) {
    marg_data_saver.reset(new basalt::MargDataSaver(marg_data_path));
    vio->out_marg_queue = &marg_data_saver->in_marg_queue;
  }

  // Mutually exclusive with --marg-data for now -- out_marg_queue only has
  // one consumer slot (see plan's Phase 3 note).
  if (online_loop_closure_enabled) {
    if (marg_data_saver) {
      std::cerr << "--online-loop-closure and --marg-data are mutually "
                   "exclusive (both want out_marg_queue). Ignoring "
                   "--online-loop-closure."
                << std::endl;
    } else {
      online_loop_closure.reset(
          new basalt::OnlineLoopClosure(calib, vio_config));
      online_loop_closure->start();
      vio->out_marg_queue = &online_loop_closure->input_queue;
    }
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
        Sophus::SE3d T_w_i;
        bool have_pose = false;

        if (online_loop_closure && online_loop_closure->getLatestCorrectedPose(T_w_i)) {
          have_pose = true;
        } else {
          auto vis_data = get_curr_vis_data_snapshot();
          if (vis_data.get()) {
            T_w_i = vis_data->states.back();
            have_pose = true;
          }
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
  }

  oakd_device->stop();
  vio->maybe_join();
  terminate = true;

  if (online_loop_closure) online_loop_closure->stop();

  // Push nullptr to output queues to unblock waiting threads
  out_vis_queue.push(nullptr);
  out_state_queue.push(nullptr);

  if (t3.get()) t3->join();
  t4.join();
  if (t5.get()) t5->join();

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
      if (online_loop_closure->getLatestCorrectedPose(T_corrected)) {
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
    for (const auto& p : vis_data->states)
      for (const auto& t_i_c : calib.T_i_c)
        render_camera((p * t_i_c).matrix(), 2.0f, state_color, 0.1f);

    for (const auto& p : vis_data->frames)
      for (const auto& t_i_c : calib.T_i_c)
        render_camera((p * t_i_c).matrix(), 2.0f, pose_color, 0.1f);

    for (const auto& t_i_c : calib.T_i_c)
      render_camera((vis_data->states.back() * t_i_c).matrix(), 2.0f, cam_color,
                    0.1f);

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
