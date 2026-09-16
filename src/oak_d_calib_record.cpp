// Records a stereo+IMU dataset from the OAK-D Lite in the EuRoC directory
// layout basalt_calibrate/basalt_calibrate_imu expect (--dataset-type
// euroc): mav0/cam0/data/<t_ns>.png + mav0/cam0/data.csv, same for cam1,
// and mav0/imu0/data.csv. Reuses OakDDevice as-is (same driver oak_d_vio.cpp
// uses for live VIO) -- this tool just drains its queues to disk instead of
// feeding an estimator. Mirrors the EuRoC layout rs_t265_record.cpp already
// writes for the RealSense T265; there was no OAK-D equivalent yet.
//
// Shows both raw camera feeds live (Pangolin, mirroring oak_d_vio.cpp's
// ImageView setup) -- added after a first blind recording (no preview)
// produced a dataset where basalt_calibrate found essentially zero valid
// poses in either camera; seeing both feeds live is the direct way to rule
// out "the grid drifted out of one camera's frame without the operator
// noticing" before chasing any deeper explanation.
//
// Usage:
//   ./basalt_oak_d_calib_record --output-dir ~/calib_recording
//
// Ctrl+C (or close the window) to stop. Point basalt_calibrate/
// basalt_calibrate_imu at the printed dataset_<timestamp>/ directory
// afterward.

#include <atomic>
#include <csignal>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <tbb/concurrent_queue.h>

#include <CLI/CLI.hpp>

#include <pangolin/display/image_view.h>
#include <pangolin/pangolin.h>

#include <basalt/device/oak_d.h>
#include <basalt/utils/filesystem.h>

std::atomic<bool> terminate{false};
void handle_shutdown_signal(int /*signum*/) { terminate = true; }

inline std::string get_date() {
  constexpr int MAX_DATE = 64;
  char the_date[MAX_DATE] = {0};
  time_t now = time(nullptr);
  if (now != -1) {
    strftime(the_date, MAX_DATE, "%Y_%m_%d_%H_%M_%S", localtime(&now));
  }
  return std::string(the_date);
}

int main(int argc, char** argv) {
  std::signal(SIGINT, handle_shutdown_signal);
  std::signal(SIGTERM, handle_shutdown_signal);

  std::string output_dir;
  CLI::App app{"OAK-D Lite calibration-dataset recorder"};
  app.add_option("--output-dir", output_dir,
                 "Parent directory to create dataset_<timestamp>/ under.")
      ->required();

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  namespace fs = basalt::fs;
  std::string dataset_dir = output_dir + "/dataset_" + get_date() + "/";
  fs::create_directories(dataset_dir + "mav0/cam0/data/");
  fs::create_directories(dataset_dir + "mav0/cam1/data/");
  fs::create_directories(dataset_dir + "mav0/imu0/");

  std::ofstream cam_csv[2] = {
      std::ofstream(dataset_dir + "mav0/cam0/data.csv"),
      std::ofstream(dataset_dir + "mav0/cam1/data.csv"),
  };
  cam_csv[0] << "#timestamp [ns],filename\n";
  cam_csv[1] << "#timestamp [ns],filename\n";

  std::ofstream imu_csv(dataset_dir + "mav0/imu0/data.csv");
  imu_csv << "#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z\n";

  std::cout << "Recording to " << dataset_dir << std::endl;
  std::cout << "Move the AprilGrid slowly through the full field of view of "
               "both cameras (check BOTH image panes -- the grid must stay "
               "fully visible in each one, not just one), at varying "
               "distances/angles/tilts, covering the frame edges and "
               "corners too -- not just the center. Include some brief "
               "pauses and some gentle rotation for the IMU calibration "
               "pass. Ctrl+C or close the window when done."
            << std::endl;

  basalt::OakDDevice::Ptr device(new basalt::OakDDevice(false));

  tbb::concurrent_bounded_queue<basalt::OpticalFlowInput::Ptr> image_queue;
  tbb::concurrent_bounded_queue<basalt::ImuData<double>::Ptr> imu_queue;
  image_queue.set_capacity(300);
  imu_queue.set_capacity(3000);

  device->setOutputQueues(&image_queue, &imu_queue);
  device->start();

  int64_t num_frames = 0, num_imu = 0;
  int64_t last_report_frames = 0;
  auto last_report_time = std::chrono::steady_clock::now();

  // Writes one frame pair to disk (data/<t_ns>.png + a data.csv row per
  // camera) -- shared by the live loop below and the post-stop drain, so
  // nothing captured right before shutdown is silently dropped.
  auto write_frame = [&](const basalt::OpticalFlowInput::Ptr& img) {
    if (!img) return;
    for (int cam_id = 0; cam_id < 2; cam_id++) {
      if (cam_id >= (int)img->img_data.size()) break;
      auto image_raw = img->img_data[cam_id].img;
      if (!image_raw.get()) continue;

      cv::Mat image(image_raw->h, image_raw->w, CV_8U);
      uint8_t* dst = image.ptr();
      const uint16_t* src = image_raw->ptr;
      for (size_t i = 0; i < image_raw->size(); i++) dst[i] = (src[i] >> 8);

      std::string filename = std::to_string(img->t_ns) + ".png";
      cv::imwrite(
          dataset_dir + "mav0/cam" + std::to_string(cam_id) + "/data/" + filename,
          image);
      cam_csv[cam_id] << img->t_ns << "," << filename << "\n";
    }
    num_frames++;
  };

  auto write_imu = [&](const basalt::ImuData<double>::Ptr& imu) {
    if (!imu) return;
    imu_csv << imu->t_ns << "," << imu->gyro[0] << "," << imu->gyro[1] << ","
            << imu->gyro[2] << "," << imu->accel[0] << "," << imu->accel[1]
            << "," << imu->accel[2] << "\n";
    num_imu++;
  };

  pangolin::CreateWindowAndBind("OAK-D Lite Calibration Recorder", 1600, 700);
  glEnable(GL_DEPTH_TEST);

  pangolin::View& img_view_display = pangolin::CreateDisplay()
                                         .SetBounds(0.0, 1.0, 0.0, 1.0)
                                         .SetLayout(pangolin::LayoutEqual);

  std::vector<std::shared_ptr<pangolin::ImageView>> img_view;
  for (int i = 0; i < 2; i++) {
    std::shared_ptr<pangolin::ImageView> iv(new pangolin::ImageView);
    img_view.push_back(iv);
    img_view_display.AddDisplay(*iv);
  }

  pangolin::GlPixFormat fmt;
  fmt.glformat = GL_LUMINANCE;
  fmt.gltype = GL_UNSIGNED_SHORT;
  fmt.scalable_internal_format = GL_LUMINANCE16;

  while (!terminate && !pangolin::ShouldQuit()) {
    basalt::OpticalFlowInput::Ptr img;
    basalt::OpticalFlowInput::Ptr latest_img;
    while (image_queue.try_pop(img)) {
      write_frame(img);
      latest_img = img;  // only the newest pair needs to be displayed
    }

    basalt::ImuData<double>::Ptr imu;
    while (imu_queue.try_pop(imu)) write_imu(imu);

    auto now = std::chrono::steady_clock::now();
    if (now - last_report_time > std::chrono::seconds(2)) {
      double fps = (num_frames - last_report_frames) /
                   std::chrono::duration<double>(now - last_report_time).count();
      std::cout << "frames=" << num_frames << " (" << fps << " fps)  imu="
                << num_imu << std::endl;
      last_report_frames = num_frames;
      last_report_time = now;
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    img_view_display.Activate();
    if (latest_img) {
      for (int cam_id = 0; cam_id < 2; cam_id++) {
        if (cam_id >= (int)latest_img->img_data.size()) break;
        auto image_raw = latest_img->img_data[cam_id].img;
        if (image_raw.get()) {
          img_view[cam_id]->SetImage(image_raw->ptr, image_raw->w,
                                     image_raw->h, image_raw->pitch, fmt);
        }
      }
    }
    pangolin::FinishFrame();

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  device->stop();

  // Drain anything left in the queues after stop() before closing files.
  basalt::OpticalFlowInput::Ptr img;
  while (image_queue.try_pop(img)) write_frame(img);
  basalt::ImuData<double>::Ptr imu;
  while (imu_queue.try_pop(imu)) write_imu(imu);

  cam_csv[0].close();
  cam_csv[1].close();
  imu_csv.close();

  std::cout << "Done. " << num_frames << " frame pairs, " << num_imu
            << " IMU samples written to " << dataset_dir << std::endl;

  return 0;
}
