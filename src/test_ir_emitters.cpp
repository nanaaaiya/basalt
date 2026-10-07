// Isolated test: do the OAK-D Pro W's IR laser dot projector / IR flood
// light intensity calls crash the firmware? Same motivation as
// test_camera_exposure.cpp -- oak_d.cpp avoids dai::CameraControl
// queries after one previously crashed the OAK-D Lite's firmware (see
// its comments), and setIrLaserDotProjectorIntensity()/
// setIrFloodLightIntensity() are a structurally different, narrower API
// (plain intensity setters on dai::Device, not CameraControl messages),
// but no Luxonis documentation confirms they're crash-safe either. This
// tool tests ONLY those two calls, across their full range, in complete
// isolation from the VIO pipeline, before either is ever allowed near
// oak_d.cpp.
//
// Live-verified safe on a real OAK-D-PRO-W (2026-09-29): both calls
// succeeded across 0.0-1.0 intensity, device stayed responsive
// afterward. This tool exists so that verification is repeatable (e.g.
// after a firmware update) rather than a one-off ad hoc check.
//
// Also captures a few frames per intensity level so the IR dot pattern's
// effect on brightness can be confirmed visually, not just "the call
// didn't throw" -- an API that silently no-ops would otherwise look
// identical to one that's actually working.
//
// Usage: ./basalt_test_ir_emitters <output_dir>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include <depthai/depthai.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

double meanBrightness(const std::shared_ptr<dai::ImgFrame>& frame) {
  cv::Mat img = frame->getCvFrame();
  cv::Mat gray;
  if (img.channels() == 3) {
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = img;
  }
  return cv::mean(gray)[0];
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <output_dir> [exposure_us iso]" << std::endl;
    return 1;
  }
  std::string output_dir = argv[1];
  std::filesystem::create_directories(output_dir);

  dai::Pipeline pipeline;

  // Same minimal, known-safe construction pattern as oak_d.cpp/
  // test_camera_exposure.cpp -- no extra requestOutput() args.
  constexpr float kCamFps = 30.0f;
  auto camLeft = pipeline.create<dai::node::Camera>()->build(
      dai::CameraBoardSocket::CAM_B, std::nullopt, kCamFps);
  // Optional fixed exposure: with auto-exposure on, the camera compensates
  // for (and drifts against) the emitters, hiding how much light they add.
  const bool manual = argc >= 4;
  if (manual) {
    camLeft->initialControl.setManualExposure(std::stoi(argv[2]), std::stoi(argv[3]));
  }
  auto* leftOut = camLeft->requestOutput(std::make_pair(640u, 480u));
  auto qLeft = leftOut->createOutputQueue(8, false);

  std::cout << "Starting pipeline..." << std::endl;
  pipeline.start();
  std::cout << "[OK] Pipeline started." << std::endl;

  dai::Device* device = pipeline.getDefaultDevice().get();
  std::cout << "Device: " << device->getDeviceName() << std::endl;

  std::ofstream csv(output_dir + "/ir_test.csv");
  csv << "phase,intensity,mean_brightness\n";

  auto captureAt = [&](const char* phase, float intensity, int num_frames) {
    double sum = 0;
    int n = 0;
    // Give the emitter a moment to actually change state before sampling.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    while (n < num_frames && pipeline.isRunning()) {
      auto frame = qLeft->get<dai::ImgFrame>();
      if (!frame) continue;
      double b = meanBrightness(frame);
      if (n == 0 && manual) {
        std::cout << "    exposure "
                  << std::chrono::duration_cast<std::chrono::microseconds>(frame->getExposureTime()).count()
                  << " us, ISO " << frame->getSensitivity() << std::endl;
      }
      sum += b;
      n++;
    }
    double mean = n > 0 ? sum / n : -1.0;
    std::cout << phase << " intensity=" << intensity
              << " mean_brightness=" << mean << " (" << n << " frames)"
              << std::endl;
    csv << phase << "," << intensity << "," << mean << "\n";
  };

  captureAt("baseline", 0.0f, 15);

  if (manual) {
    // Alternate off/on so any remaining drift shows up as off-to-off change.
    std::cout << "\nFixed exposure: alternating emitters off/on..." << std::endl;
    for (float intensity : {0.0f, 0.17f, 0.0f, 0.17f, 0.0f, 0.5f, 0.0f, 1.0f, 0.0f}) {
      device->setIrLaserDotProjectorIntensity(intensity);
      captureAt("laser", intensity, 15);
    }
    for (float intensity : {0.0f, 0.5f, 0.0f, 1.0f, 0.0f}) {
      device->setIrFloodLightIntensity(intensity);
      captureAt("flood", intensity, 15);
    }
    device->setIrFloodLightIntensity(0.0f);
    std::cout << "\n[OK] Device still responsive: " << device->getDeviceName() << std::endl;
    return 0;
  }

  std::cout << "\nTesting IR laser dot projector intensity..." << std::endl;
  for (float intensity : {0.0f, 0.3f, 0.6f, 1.0f, 0.0f}) {
    bool ok = device->setIrLaserDotProjectorIntensity(intensity);
    std::cout << "  setIrLaserDotProjectorIntensity(" << intensity
              << ") returned " << ok << std::endl;
    captureAt("laser", intensity, 15);
  }

  std::cout << "\nTesting IR flood light intensity..." << std::endl;
  for (float intensity : {0.0f, 0.3f, 0.6f, 1.0f, 0.0f}) {
    bool ok = device->setIrFloodLightIntensity(intensity);
    std::cout << "  setIrFloodLightIntensity(" << intensity << ") returned "
              << ok << std::endl;
    captureAt("flood", intensity, 15);
  }

  std::cout << "\n[OK] Device still responsive: " << device->getDeviceName()
            << std::endl;

  pipeline.stop();
  pipeline.wait();
  std::cout << "[OK] Pipeline stopped cleanly." << std::endl;
  return 0;
}
