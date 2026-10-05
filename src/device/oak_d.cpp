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

#include <basalt/device/oak_d.h>

#include <chrono>
#include <iostream>
#include <optional>
#include <utility>

#include <opencv2/core.hpp>

namespace basalt {

namespace {

double to_seconds(
    std::chrono::time_point<std::chrono::steady_clock,
                             std::chrono::steady_clock::duration>
        tp) {
  return std::chrono::duration<double>(tp.time_since_epoch()).count();
}

}  // namespace

OakDDevice::OakDDevice(bool enable_stereo_depth, bool depth_full_res)
    : enable_stereo_depth_(enable_stereo_depth),
      depth_full_res_(depth_full_res) {}

OakDDevice::~OakDDevice() { stop(); }

void OakDDevice::start() {
  if (running.exchange(true)) return;

  // Two raw (unrectified) Camera outputs (left=CAM_B, right=CAM_C) + raw
  // 6-axis IMU, at the mono sensors' native 640x480 -- must match the
  // resolution baked into the calibration file passed via --cam-calib.
  // See run_oakd.cpp (open_vins project) for why requestOutput() is called
  // with only the size argument: passing explicit GRAY8/CROP/fps/
  // enableUndistortion args here previously triggered a real on-device
  // firmware crash (PlgSrcMipi rejecting the config). The mono sensors
  // auto-select GRAY8 and default undistortion is off, so omitting the extra
  // args keeps raw, unrectified frames without losing anything we need.
  auto camLeft = pipeline.create<dai::node::Camera>()->build(
      dai::CameraBoardSocket::CAM_B, std::nullopt, (float)CAM_FPS);
  auto camRight = pipeline.create<dai::node::Camera>()->build(
      dai::CameraBoardSocket::CAM_C, std::nullopt, (float)CAM_FPS);

  // See MAX_EXPOSURE_US's header comment: caps auto-exposure's max
  // exposure time to reduce motion blur, verified safe via
  // basalt_test_camera_exposure before ever being added here. Set via
  // initialControl (applied once, before pipeline.start() below) rather
  // than a runtime inputControl queue message, matching what was actually
  // tested.
  camLeft->initialControl.setAutoExposureLimit(
      std::chrono::microseconds(MAX_EXPOSURE_US));
  camRight->initialControl.setAutoExposureLimit(
      std::chrono::microseconds(MAX_EXPOSURE_US));

  auto* leftOut = camLeft->requestOutput(std::make_pair(640u, 480u));
  auto* rightOut = camRight->requestOutput(std::make_pair(640u, 480u));

  auto imu = pipeline.create<dai::node::IMU>();
  imu->enableIMUSensor(
      {dai::IMUSensor::ACCELEROMETER_RAW, dai::IMUSensor::GYROSCOPE_RAW},
      IMU_RATE);
  imu->setBatchReportThreshold(1);
  imu->setMaxBatchReports(10);

  q_left = leftOut->createOutputQueue(8, false);
  q_right = rightOut->createOutputQueue(8, false);
  q_imu = imu->out.createOutputQueue(50, false);

  if (enable_stereo_depth_) {
    // For the occupancy-grid mapper (not VIO -- that still only uses the
    // raw mono frames above). StereoDepth rectifies leftOut/rightOut
    // internally using the device's calibration, so it's fine that
    // they're the same raw, unrectified outputs VIO also reads.
    auto stereo = pipeline.create<dai::node::StereoDepth>();
    stereo->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::DEFAULT);
    // Untouched on the OAK-D Lite (bare DEFAULT preset, no filtering) --
    // worth enabling now since this is a fresh integration anyway and
    // active depth (OAK-D Pro W's IR laser projector, see setIrEmitters())
    // should make left-right consistency checking and subpixel refinement
    // meaningfully more effective than they'd have been on pure passive
    // stereo. Confidence threshold left near DepthAI's own default (55) --
    // an explicit first guess, not yet live-tuned against this specific
    // camera's actual noise floor.
    stereo->setLeftRightCheck(true);
    stereo->setSubpixel(true);
    // depthai v3: disparities with confidence OVER this threshold are kept,
    // so a higher value is stricter. 180 left only ~9% of a wall with depth
    // in a real scan (and 39% vs 65% at 55 on a static bench test), far too
    // sparse for surface fusion. 55 is depthai's own default.
    stereo->initialConfig->setConfidenceThreshold(55);
    // Matches depthai-core's own examples/python/StereoDepth/stereo.py,
    // confirmed by the user to produce a visibly correct depth map on
    // this exact device -- that example enables this and we didn't.
    // Extends the disparity search range, which mainly improves NEAR-
    // range accuracy -- relevant since this rig's real test distance
    // (~0.4-0.5m) is close enough that plain (non-extended) disparity
    // range is a plausible contributor to the bad-depth-value pattern
    // diagnosed 2026-09-29 (min/max depth filter, IR intensity).
    stereo->setExtendedDisparity(true);
    // The DEFAULT preset's temporal filter blends each pixel's disparity
    // with previous frames, which assumes a still camera -- while panning
    // it smears surfaces from earlier viewpoints into the current frame.
    stereo->initialConfig->postProcessing.temporalFilter.enable = false;
    // CRITICAL: without this, depth alignment defaults to AUTO, which for a
    // non-RGB-aligned stereo pair resolves to the RIGHT camera's rectified
    // frame -- but depth_intrinsics_ (queried below via CAM_B) and
    // oak_d_vio.cpp's T_w_c composition (pose * calib.T_i_c[0], cam0/left's
    // extrinsic) both assume depth lives in the LEFT camera's frame. That
    // mismatch is a rigid offset roughly one baseline (~7.5cm) in a fixed
    // direction for every point -- live-diagnosed as the cause of the
    // "line of voxels swept off to one side, overlapping the frustum"
    // pattern on OAK-D Pro W, 2026-09-29 (see occupancy_mapper.h's header
    // comment for the sibling wrong-camera-model bug this compounded with).
    stereo->setDepthAlign(dai::CameraBoardSocket::CAM_B);
    // The DEFAULT preset downscales its own output resolution regardless
    // of the 640x480 mono input (found producing 320x240 depth frames in
    // testing). Deliberately left at that lower resolution rather than
    // forced up to 640x480: doing so quadruples the depth data volume over
    // USB, which reproduced this device's known connection-crash pattern
    // in live testing. OccupancyMapper::insertFrame() instead scales pixel
    // coordinates to match the calibration's resolution before
    // unprojecting, so the intrinsics stay correct at whatever resolution
    // the depth stream actually outputs -- see its comment for the fuller
    // explanation of the fan-shaped-map bug this was fixing.
    if (depth_full_res_) {
      camLeft->requestOutput(std::make_pair(1280u, 800u), std::nullopt,
                             dai::ImgResizeMode::CROP, DEPTH_FULL_RES_FPS)
          ->link(stereo->left);
      camRight->requestOutput(std::make_pair(1280u, 800u), std::nullopt,
                              dai::ImgResizeMode::CROP, DEPTH_FULL_RES_FPS)
          ->link(stereo->right);
    } else {
      leftOut->link(stereo->left);
      rightOut->link(stereo->right);
    }
    q_depth = stereo->depth.createOutputQueue(8, false);
  }

  pipeline.start();
  std::cout << "[OAKD]: device connected, streaming" << std::endl;

  if (enable_stereo_depth_) {
    // See DepthIntrinsics's header comment for why this must be Luxonis's
    // own factory calibration, not calib_'s (Basalt's own, of the RAW
    // lens). Queried at 640x480 -- OccupancyMapper's existing scale_u/
    // scale_v logic already adapts to whatever resolution the depth
    // stream actually outputs (see its own comment), same as it already
    // does for calib_'s resolution.
    auto device = pipeline.getDefaultDevice();
    if (device) {
      // requestOutput(640x480) from the 1280x800 OV9282 scales by 0.6 and
      // CROPS the width (768 -> 640), so focal length is 0.6x native, not
      // the 0.5x that getCameraIntrinsics(CAM_B, 640, 480) assumes (it
      // returned fx=282 where the real frames have 338.5 -- confirmed via
      // ImgFrame::getTransformation() on the mono stream). The 20% focal
      // error tilted every depth frame's surfaces by a view-angle-dependent
      // amount, fanning one wall into several sheets in both the live map
      // and offline TSDF.
      auto calib = device->readCalibration();
      constexpr int kSensorW = 1280, kSensorH = 800;
      const int kOutW = depth_full_res_ ? 1280 : 640;
      const int kOutH = depth_full_res_ ? 800 : 480;
      auto intr = calib.getCameraIntrinsics(dai::CameraBoardSocket::CAM_B,
                                            kSensorW, kSensorH);
      const double s = std::max(static_cast<double>(kOutW) / kSensorW,
                                static_cast<double>(kOutH) / kSensorH);
      depth_intrinsics_.fx = intr[0][0] * s;
      depth_intrinsics_.fy = intr[1][1] * s;
      depth_intrinsics_.cx = intr[0][2] * s - (kSensorW * s - kOutW) / 2.0;
      depth_intrinsics_.cy = intr[1][2] * s - (kSensorH * s - kOutH) / 2.0;
      depth_intrinsics_.width = kOutW;
      depth_intrinsics_.height = kOutH;
      std::cout << "[OAKD] Depth (rectified) intrinsics: fx="
                << depth_intrinsics_.fx << " fy=" << depth_intrinsics_.fy
                << " cx=" << depth_intrinsics_.cx
                << " cy=" << depth_intrinsics_.cy << " @ " << kOutW << "x"
                << kOutH
                << std::endl;
    } else {
      std::cerr << "[OAKD] Could not get device handle for depth "
                   "intrinsics -- occupancy mapping will be wrong"
                << std::endl;
    }
  }

  device_thread = std::thread(&OakDDevice::deviceLoop, this);
}

void OakDDevice::setIrEmitters(float laser_intensity, float flood_intensity) {
  if (!running.load()) {
    std::cerr << "[OAKD] setIrEmitters() called before start() -- no-op"
              << std::endl;
    return;
  }
  auto device = pipeline.getDefaultDevice();
  if (!device) {
    std::cerr << "[OAKD] setIrEmitters(): no device handle available"
              << std::endl;
    return;
  }
  bool laser_ok = device->setIrLaserDotProjectorIntensity(laser_intensity);
  bool flood_ok = device->setIrFloodLightIntensity(flood_intensity);
  std::cout << "[OAKD] IR laser intensity=" << laser_intensity << " ("
            << (laser_ok ? "ok" : "unsupported/failed") << "), flood intensity="
            << flood_intensity << " (" << (flood_ok ? "ok" : "unsupported/failed")
            << ")" << std::endl;
}

void OakDDevice::stop() {
  if (!running.exchange(false)) return;

  if (device_thread.joinable()) device_thread.join();

  try {
    pipeline.stop();
    pipeline.wait();
  } catch (const std::exception&) {
  }

  OutputQueues queues;
  {
    std::lock_guard<std::mutex> lock(output_queues_mutex);
    queues = output_queues;
  }
  if (queues.image_data_queue) queues.image_data_queue->push(nullptr);
  if (queues.imu_data_queue) queues.imu_data_queue->push(nullptr);
  if (queues.depth_data_queue) queues.depth_data_queue->push(nullptr);
}

void OakDDevice::deviceLoop() {
  // Mirrors run_oakd.cpp's main loop: buffer+pair stereo frames by nearest
  // timestamp, only feed a frame pair once IMU data has caught up to it
  // (VioEstimatorBase expects IMU data for a timestamp before the
  // corresponding image, same discipline OpenVINS's ROS1Visualizer uses).
  struct StampedFrame {
    double t;
    std::shared_ptr<dai::ImgFrame> frame;
  };
  std::deque<StampedFrame> left_queue, right_queue;
  double last_imu_time = -1.0;

  // DepthAI's own device reconnection (after an X_LINK_ERROR/crash) is
  // transparent to this queue-pull API -- tryGet() just silently starts
  // returning data again once reconnected, with no explicit "reconnected"
  // event exposed here. A timestamp discontinuity in the IMU stream is
  // the most direct, robust signal available instead, regardless of the
  // stall's specific cause. Normal IMU_RATE=200Hz means consecutive
  // samples are ~5ms apart, so this threshold is a huge margin above any
  // ordinary jitter.
  constexpr double kDeviceGapThresholdSeconds = 0.5;
  // How long to discard data for after a detected gap, before trusting
  // it again. Root-caused via a real bad-initialization case: the very
  // first frame available right after a device crash+reconnect had only
  // 12 corners detected (vs. a normal 300-700+), and VIO's one-shot
  // gravity-alignment step used the first available IMU sample from that
  // same window as its sole reference, producing a badly wrong initial
  // orientation (confirmed live: a near-inverted Z axis) that then
  // leaked into every subsequent pose as a steady, uncancelled-gravity
  // drift for the rest of the session. Discarding a brief settle window
  // gives the stream a chance to stabilize before anything downstream --
  // VIO's initialization above all -- ever sees it.
  constexpr double kDeviceGapSettleSeconds = 1.0;
  double discard_until = -1.0;

  while (running.load() && pipeline.isRunning()) {
    bool got_data = false;

    OutputQueues queues;
    {
      std::lock_guard<std::mutex> lock(output_queues_mutex);
      queues = output_queues;
    }

    // Independent of the IMU/stereo-frame pairing below -- the occupancy
    // mapper times its own depth frames against VIO's pose stream itself,
    // the same way DashboardClient consumes poses without needing to be
    // threaded through this pairing logic. try_push, not push: this same
    // loop also reads the IMU/mono frames VIO actually needs every single
    // one of, so a depth consumer that ever falls behind must never be
    // able to block this thread -- dropping a depth frame costs nothing
    // but a slightly staler map, dropping an IMU sample would be a real
    // VIO correctness problem.
    if (q_depth) {
      while (auto depthFrame = q_depth->tryGet<dai::ImgFrame>()) {
        got_data = true;
        if (queues.depth_data_queue) queues.depth_data_queue->try_push(depthFrame);
      }
    }

    while (auto imuData = q_imu->tryGet<dai::IMUData>()) {
      got_data = true;
      for (auto& packet : imuData->packets) {
        double t = to_seconds(packet.acceleroMeter.getTimestamp());

        if (last_imu_time >= 0 && t - last_imu_time > kDeviceGapThresholdSeconds) {
          discard_until = t + kDeviceGapSettleSeconds;
          left_queue.clear();
          right_queue.clear();
          std::cout << "[OAKD] stream gap detected (" << (t - last_imu_time)
                    << "s) -- discarding data for " << kDeviceGapSettleSeconds
                    << "s while the stream settles" << std::endl;
        }
        last_imu_time = t;

        ImuData<double>::Ptr data;
        data.reset(new ImuData<double>);
        data->t_ns = (int64_t)(t * 1e9);
        data->accel << packet.acceleroMeter.x, packet.acceleroMeter.y,
            packet.acceleroMeter.z;
        data->gyro << packet.gyroscope.x, packet.gyroscope.y,
            packet.gyroscope.z;

        // Raw tap: pushed unconditionally, even during a post-gap settle
        // window -- a gyro reading itself isn't corrupted by a stream
        // gap the way vision-derived data is, so a consumer that only
        // wants rotation-rate (not VIO's own fragile init) shouldn't be
        // starved by VIO-specific discard logic.
        if (queues.imu_tap_queue) queues.imu_tap_queue->try_push(data);

        if (t < discard_until) continue;  // still settling -- drop this sample

        if (queues.imu_data_queue) queues.imu_data_queue->push(data);
      }
    }

    while (auto frame = q_left->tryGet<dai::ImgFrame>()) {
      got_data = true;
      double t = to_seconds(frame->getTimestamp());
      if (t < discard_until) continue;  // still settling after a stream gap
      left_queue.push_back({t, frame});
    }
    while (auto frame = q_right->tryGet<dai::ImgFrame>()) {
      got_data = true;
      double t = to_seconds(frame->getTimestamp());
      if (t < discard_until) continue;
      right_queue.push_back({t, frame});
    }

    while (!left_queue.empty() && !right_queue.empty()) {
      double dt = left_queue.front().t - right_queue.front().t;
      if (std::abs(dt) > 1.0 / (2.0 * CAM_FPS)) {
        if (dt < 0) {
          left_queue.pop_front();
        } else {
          right_queue.pop_front();
        }
        continue;
      }
      double t = 0.5 * (left_queue.front().t + right_queue.front().t);
      if (last_imu_time < 0 || t > last_imu_time) {
        break;  // wait for more IMU data before we feed this frame pair
      }

      OpticalFlowInput::Ptr data(new OpticalFlowInput);
      data->img_data.resize(NUM_CAMS);
      data->t_ns = (int64_t)(t * 1e9);

      std::shared_ptr<dai::ImgFrame> frames[NUM_CAMS] = {
          left_queue.front().frame, right_queue.front().frame};

      for (int i = 0; i < NUM_CAMS; i++) {
        cv::Mat img = frames[i]->getCvFrame();

        data->img_data[i].img.reset(
            new ManagedImage<uint16_t>(img.cols, img.rows));

        const uint8_t* data_in = img.ptr<uint8_t>(0);
        uint16_t* data_out = data->img_data[i].img->ptr;

        size_t full_size = (size_t)img.cols * (size_t)img.rows;

        // A "low light" signal was deliberately never built here via
        // dai::CameraControl exposure/gain queries -- that path
        // previously crashed the OAK-D firmware (see the pipeline setup
        // above). This computes mean brightness directly from pixel data
        // already being read for the bit-shift conversion below (folded
        // into the same pass, not a second scan) -- zero camera-control
        // calls, cam0/left only (matches this file's existing
        // diagnostic convention of treating cam0 as the reference).
        uint64_t brightness_sum = 0;

        for (size_t j = 0; j < full_size; j++) {
          int val = data_in[j];
          if (i == 0) brightness_sum += (unsigned)val;
          val = val << 8;
          data_out[j] = val;
        }

        if (i == 0 && full_size > 0) {
          latest_cam0_mean_brightness =
              (double)brightness_sum / (double)full_size;
        }
      }

      {
        std::lock_guard<std::mutex> lock(last_img_data_mutex);
        last_img_data = data;
      }
      if (queues.image_data_queue) queues.image_data_queue->push(data);

      left_queue.pop_front();
      right_queue.pop_front();
    }

    if (!got_data) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
}

void OakDDevice::setOutputQueues(
    tbb::concurrent_bounded_queue<OpticalFlowInput::Ptr>* image_queue,
    tbb::concurrent_bounded_queue<ImuData<double>::Ptr>* imu_queue) {
  std::lock_guard<std::mutex> lock(output_queues_mutex);
  output_queues.image_data_queue = image_queue;
  output_queues.imu_data_queue = imu_queue;
}

void OakDDevice::setDepthOutputQueue(
    tbb::concurrent_bounded_queue<std::shared_ptr<dai::ImgFrame>>*
        depth_queue) {
  std::lock_guard<std::mutex> lock(output_queues_mutex);
  output_queues.depth_data_queue = depth_queue;
}

void OakDDevice::setImuTapQueue(
    tbb::concurrent_bounded_queue<ImuData<double>::Ptr>* imu_tap_queue) {
  std::lock_guard<std::mutex> lock(output_queues_mutex);
  output_queues.imu_tap_queue = imu_tap_queue;
}

void OakDDevice::detachOutputQueues() {
  std::lock_guard<std::mutex> lock(output_queues_mutex);
  output_queues = {};
}

OpticalFlowInput::Ptr OakDDevice::getLastImageData() const {
  std::lock_guard<std::mutex> lock(last_img_data_mutex);
  return last_img_data;
}

}  // namespace basalt
