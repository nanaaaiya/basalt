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

#include <basalt/calibration/cam_calib.h>

#include <CLI/CLI.hpp>

int main(int argc, char** argv) {
  std::string dataset_path;
  std::string dataset_type;
  std::string aprilgrid_path;
  std::string result_path;
  std::vector<std::string> cam_types;
  std::string cache_dataset_name = "calib-cam";
  int skip_images = 1;
  bool no_gui = false;

  CLI::App app{"Calibrate IMU"};

  app.add_option("--dataset-path", dataset_path, "Path to dataset")->required();
  app.add_option("--result-path", result_path, "Path to result folder")
      ->required();
  app.add_option("--dataset-type", dataset_type, "Dataset type (euroc, bag)")
      ->required();

  app.add_option("--aprilgrid", aprilgrid_path,
                 "Path to Aprilgrid config file)")
      ->required();

  app.add_option("--cache-name", cache_dataset_name,
                 "Name to save cached files");

  app.add_option("--skip-images", skip_images, "Number of images to skip");
  app.add_option("--cam-types", cam_types,
                 "Type of cameras (eucm, ds, kb4, pinhole)")
      ->required();
  app.add_flag("--no-gui", no_gui, "Run calibration without opening the GUI");

  bool skip_vign = false;
  app.add_flag("--skip-vign", skip_vign,
              "Skip compute_vign() in the --no-gui path. Vignette estimation "
              "requires a target with dedicated vignette-sampling markers "
              "(printed by Kalibr's default AprilGrid PDF, absent from a "
              "plain AprilTag grid) and a static-target/moving-camera "
              "capture with constant lighting -- on data that doesn't meet "
              "that, it has been observed to crash with an out-of-bounds "
              "RdSpline assertion. Harmless to skip whenever vignette "
              "calibration isn't actually needed.");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  basalt::CamCalib cv(dataset_path, dataset_type, aprilgrid_path, result_path,
                      cache_dataset_name, skip_images, cam_types, !no_gui);

  if (no_gui) {
    cv.loadDataset();
    cv.detectCorners();
    cv.initCamIntrinsics();
    cv.initCamPoses();
    cv.initCamExtrinsics();
    cv.initOptimization();

    // Two-phase optimization, per doc/Calibration.md's own documented
    // remedy for exactly this failure mode ("opt_intr controls if the
    // optimization can change the intrinsics. For some datasets it might
    // be helpful to disable this option for several first iterations"):
    // on a real OAK-D Lite handheld capture, letting intrinsics move
    // jointly with poses/extrinsics from the very first iteration let the
    // optimizer wander from a good, known-correct intrinsics seed to a
    // converged-but-wrong one (~31px mean reprojection error, fx drifting
    // from the seeded ~461 to ~630-760). Locking intrinsics first gives
    // poses/extrinsics a chance to settle around the good seed before
    // intrinsics are allowed to move at all.
    std::cout << "Phase 1: optimizing poses/extrinsics with intrinsics "
                 "locked at their seed values..."
              << std::endl;
    cv.setOptIntrinsics(false);
    for (int i = 0; i < 30 && !cv.optimizeWithParam(true); i++) {
    }

    std::cout << "Phase 2: joint optimization with intrinsics unlocked..."
              << std::endl;
    cv.setOptIntrinsics(true);
    while (!cv.optimizeWithParam(true)) {
    }

    if (!skip_vign) cv.computeVign();
    cv.saveCalib();
    return 0;
  }

  cv.renderingLoop();

  return 0;
}
