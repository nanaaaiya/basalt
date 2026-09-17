// Isolated test: does capping the OAK-D Lite's auto-exposure maximum
// exposure time crash the firmware? oak_d.cpp deliberately never touches
// dai::CameraControl beyond the pipeline's implicit defaults, after
// QUERYING exposure/gain state previously crashed the firmware (see its
// comments) -- but that was specifically a read (asking the camera for
// its current auto-exposure values), not a write (telling the camera what
// exposure to use), which is a different and far more commonly-used
// DepthAI code path. This tool tests ONLY the write path, in complete
// isolation from the VIO pipeline, before it's ever allowed near
// oak_d.cpp.
//
// setAutoExposureLimit() is deliberately used here instead of full manual
// exposure (setManualExposure()): auto-exposure keeps adapting to
// lighting, it's just barred from ever choosing an exposure time long
// enough to blur under real motion. Set via initialControl (applied once,
// before the pipeline starts) rather than a runtime inputControl queue
// message, since the documented past crash was specifically from a
// runtime control/query path -- this stays as close as possible to a
// one-time configuration change instead of a live control loop.
//
// Reports brightness and Laplacian-variance sharpness per frame so the
// effect can be measured directly against the already-established
// baselines: ~700-800 stationary, 157 (worst) to 1215 (best) handheld
// motion with auto-exposure uncapped, median 469.
//
// Usage: ./basalt_test_camera_exposure <output_dir> [max_exposure_us] [num_frames]

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <depthai/depthai.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " <output_dir> [max_exposure_us=8000] [num_frames=90]"
              << std::endl;
    return 1;
  }
  std::string output_dir = argv[1];
  int max_exposure_us = argc >= 3 ? std::stoi(argv[2]) : 8000;
  int num_frames = argc >= 4 ? std::stoi(argv[3]) : 90;

  std::filesystem::create_directories(output_dir);

  dai::Pipeline pipeline;

  // Same minimal, known-safe construction pattern as oak_d.cpp: no extra
  // args to requestOutput() (passing GRAY8/CROP/fps/enableUndistortion
  // there previously crashed the firmware with "PlgSrcMipi rejecting the
  // config" -- a separate, already-known issue, avoided here anyway to
  // keep this test isolated to ONLY the exposure question).
  constexpr float kCamFps = 30.0f;
  auto camLeft = pipeline.create<dai::node::Camera>()->build(
      dai::CameraBoardSocket::CAM_B, std::nullopt, kCamFps);
  auto camRight = pipeline.create<dai::node::Camera>()->build(
      dai::CameraBoardSocket::CAM_C, std::nullopt, kCamFps);

  std::cout << "Setting auto-exposure max exposure time to "
            << max_exposure_us << "us on both cameras..." << std::endl;
  camLeft->initialControl.setAutoExposureLimit(
      std::chrono::microseconds(max_exposure_us));
  camRight->initialControl.setAutoExposureLimit(
      std::chrono::microseconds(max_exposure_us));

  auto* leftOut = camLeft->requestOutput(std::make_pair(640u, 480u));
  auto* rightOut = camRight->requestOutput(std::make_pair(640u, 480u));

  auto qLeft = leftOut->createOutputQueue(8, false);
  auto qRight = rightOut->createOutputQueue(8, false);

  std::cout << "Starting pipeline..." << std::endl;
  pipeline.start();
  std::cout << "[OK] Pipeline started without crashing." << std::endl;

  std::ofstream csv(output_dir + "/frames.csv");
  csv << "frame_idx,brightness,laplacian_var,exposure_us,saved_filename\n";

  int saved = 0;
  int frame_idx = 0;
  auto start = std::chrono::steady_clock::now();
  double sum_brightness = 0, sum_var = 0;

  while (frame_idx < num_frames && pipeline.isRunning()) {
    auto frame = qLeft->get<dai::ImgFrame>();
    if (!frame) continue;
    cv::Mat img = frame->getCvFrame();
    if (img.empty()) continue;

    cv::Mat gray;
    if (img.channels() == 3) {
      cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
      gray = img;
    }

    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(lap, mean, stddev);
    double variance = stddev[0] * stddev[0];
    double brightness = cv::mean(gray)[0];
    sum_brightness += brightness;
    sum_var += variance;

    // getExposureTime() reads metadata already attached to this frame by
    // the device -- not a live query back to the camera (the operation
    // documented as crashing before) -- so this is safe, and lets us
    // directly confirm the requested cap is actually being honored
    // instead of inferring it indirectly from sharpness alone.
    auto exposure_us = frame->getExposureTime();
    std::cout << "frame " << frame_idx << ": brightness=" << brightness
              << " laplacian_var=" << variance
              << " exposure_us=" << exposure_us.count() << std::endl;

    std::string saved_filename;
    if (frame_idx % 5 == 0) {
      char buf[256];
      std::snprintf(buf, sizeof(buf), "frame_%04d.png", frame_idx);
      saved_filename = buf;
      cv::imwrite(output_dir + "/" + saved_filename, gray);
      saved++;
    }
    csv << frame_idx << "," << brightness << "," << variance << ","
        << exposure_us.count() << "," << saved_filename << "\n";

    // Drain the right queue too so it doesn't back up -- not analyzed,
    // this test only needs one camera's stream to answer the crash/blur
    // question.
    while (qRight->tryGet<dai::ImgFrame>()) {
    }

    frame_idx++;
  }

  double elapsed = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - start)
                        .count();
  std::cout << "\nCaptured " << frame_idx << " frames in " << elapsed
            << "s, saved " << saved << " to " << output_dir << std::endl;
  if (frame_idx > 0) {
    std::cout << "mean brightness=" << (sum_brightness / frame_idx)
              << " mean laplacian_var=" << (sum_var / frame_idx) << std::endl;
  }

  pipeline.stop();
  pipeline.wait();
  std::cout << "[OK] Pipeline stopped cleanly." << std::endl;
  return 0;
}
