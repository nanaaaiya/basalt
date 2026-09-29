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

// Live driver for a Luxonis OAK-D camera (stereo mono cameras + IMU). This
// branch (oakd-pro) targets the OAK-D Pro W specifically: OV9282 wide-FOV
// global-shutter mono pair, active depth via IR laser dot projector (see
// setIrEmitters() below), and a BNO086 9-axis IMU -- confirmed by direct
// query (device.getConnectedIMU()) on real hardware 2026-09-29, NOT the
// BMI270 the OAK-D Lite (this driver's original target) uses. IMU_RATE=200
// with ACCELEROMETER_RAW/GYROSCOPE_RAW reports (see enableIMUSensor() in
// the .cpp) is unchanged and live-verified to still work on the BNO086
// (~191Hz measured, close enough to the 200Hz request to match the Lite's
// own real-world jitter) -- DepthAI's raw IMU report types are chip-
// agnostic at the API level, so no code change was needed there, only
// this comment. Actual noise characteristics (bias stability, noise
// density) differ from BMI270's and are captured fresh by
// basalt_calibrate_imu regardless of which chip is behind the API, same
// as for any new physical unit.
// Modeled directly on RsT265Device's public shape (start()/stop()/
// setOutputQueues()) so it drops into the same OpticalFlowBase::input_queue /
// VioEstimatorBase::imu_data_queue wiring used by rs_t265_vio.cpp. Unlike the
// T265, the OAK-D has no on-device factory calibration to query at runtime --
// calibration is always supplied externally (see oak_d_vio.cpp's --cam-calib),
// so there is no exportCalibration() here.
//
// DepthAI's v3 API is queue-pull-based (tryGet), not callback-push like
// librealsense, so start() spins up its own background thread that mirrors
// run_oakd.cpp's main loop (drain IMU, buffer+pair stereo frames by nearest
// timestamp, only feed a frame pair once IMU data has caught up to it).

#pragma once

#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

#include <depthai/depthai.hpp>

#include <tbb/concurrent_queue.h>

#include <basalt/imu/imu_types.h>
#include <basalt/optical_flow/optical_flow.h>

namespace basalt {

class OakDDevice {
 public:
  using Ptr = std::shared_ptr<OakDDevice>;

  static constexpr int IMU_RATE = 200;
  static constexpr int CAM_FPS = 30;
  static constexpr int NUM_CAMS = 2;

  // Caps auto-exposure's maximum exposure time (still adapts to lighting,
  // just never chooses an exposure long enough to blur under real motion)
  // -- set via dai::CameraControl::setAutoExposureLimit() in start(). This
  // is a WRITE to the camera (telling it what to do), not a QUERY (asking
  // it for its current state) -- the latter is documented elsewhere in
  // this file as having previously crashed the OAK-D firmware; the former
  // was verified safe first, in complete isolation from this pipeline,
  // via basalt_test_camera_exposure (see that tool's own comment).
  // Confirmed on real handheld footage: 8ms holds the camera to
  // Laplacian-variance sharpness in the same range as genuinely
  // stationary footage during moderate motion, only degrading on the
  // fastest deliberate whips -- a real, measured improvement over
  // uncapped auto-exposure's much wider blur range, not a full fix for
  // arbitrarily fast motion.
  static constexpr int MAX_EXPOSURE_US = 8000;

  // enable_stereo_depth: builds and runs the on-device StereoDepth node
  // (for the future occupancy-grid mapper) alongside the existing raw
  // mono + IMU streams VIO uses. This has to be a constructor-time
  // choice, not a setter called after start() like setOutputQueues()
  // below -- DepthAI's node graph is fixed once pipeline.start() runs,
  // so whether the StereoDepth node exists at all can't be decided
  // later. Defaults to false so every existing caller (just
  // `new OakDDevice`) keeps paying zero extra device-side compute for a
  // stream it never asked for.
  explicit OakDDevice(bool enable_stereo_depth = false);
  ~OakDDevice();

  void start();
  void stop();

  void setOutputQueues(
      tbb::concurrent_bounded_queue<OpticalFlowInput::Ptr>* image_queue,
      tbb::concurrent_bounded_queue<ImuData<double>::Ptr>* imu_queue);
  void detachOutputQueues();

  // Separate from setOutputQueues() above -- depth is for the (future)
  // occupancy-grid mapper, an entirely separate consumer from VIO's
  // image/IMU wiring, and shouldn't need to touch that call at every
  // existing call site just to add this. Safe to call whether or not
  // enable_stereo_depth was set; it's simply never fed if not.
  void setDepthOutputQueue(
      tbb::concurrent_bounded_queue<std::shared_ptr<dai::ImgFrame>>*
          depth_queue);

  // A raw, unconsumed tap of every IMU sample, independent of
  // setOutputQueues()'s imu_queue above (which VIO drains/consumes for
  // preintegration). Exists because gyro data currently dies inside the
  // VIO estimator once used -- nothing downstream of it can query
  // "how fast is this rotating right now" today. Anything that wants raw
  // IMU regardless of whether VIO has even initialized yet (a future
  // fusion layer, scenario-characterization tooling) should read from
  // here instead of trying to intercept setOutputQueues()'s queue.
  void setImuTapQueue(
      tbb::concurrent_bounded_queue<ImuData<double>::Ptr>* imu_tap_queue);

  // IR laser dot projector / IR flood light intensity, OAK-D Pro W only
  // (both silently no-ops on hardware without them -- DepthAI's own
  // setIr*Intensity() return false rather than throwing when unsupported,
  // per live testing). 0.0 = off, 1.0 = maximum. A plain runtime call on
  // dai::Device, NOT a dai::CameraControl message -- see this class's
  // header comment on why that distinction matters here. Live-verified
  // safe (no firmware crash, full 0.0-1.0 range, both laser and flood)
  // via basalt_test_ir_emitters before this was ever wired in here.
  // Must be called after start() (pipeline.getDefaultDevice() is only
  // valid once the pipeline has actually started) -- a no-op before
  // that, logged rather than silently dropped.
  void setIrEmitters(float laser_intensity, float flood_intensity);

  OpticalFlowInput::Ptr getLastImageData() const;

  // Mean pixel intensity (0-255) of the most recent cam0/left frame --
  // a "low light" proxy computed directly from pixel data, deliberately
  // NOT via dai::CameraControl exposure/gain queries (see deviceLoop()).
  double getLatestCam0MeanBrightness() const {
    return latest_cam0_mean_brightness;
  }

 private:
  void deviceLoop();

  const bool enable_stereo_depth_;

  std::atomic<bool> running{false};
  std::thread device_thread;

  dai::Pipeline pipeline;
  std::shared_ptr<dai::MessageQueue> q_left;
  std::shared_ptr<dai::MessageQueue> q_right;
  std::shared_ptr<dai::MessageQueue> q_imu;
  std::shared_ptr<dai::MessageQueue> q_depth;  // null unless enable_stereo_depth_

  mutable std::mutex last_img_data_mutex;
  OpticalFlowInput::Ptr last_img_data;

  std::atomic<double> latest_cam0_mean_brightness{255.0};

  struct OutputQueues {
    tbb::concurrent_bounded_queue<OpticalFlowInput::Ptr>* image_data_queue =
        nullptr;
    tbb::concurrent_bounded_queue<ImuData<double>::Ptr>* imu_data_queue =
        nullptr;
    tbb::concurrent_bounded_queue<std::shared_ptr<dai::ImgFrame>>*
        depth_data_queue = nullptr;
    tbb::concurrent_bounded_queue<ImuData<double>::Ptr>* imu_tap_queue =
        nullptr;
  };

  mutable std::mutex output_queues_mutex;
  OutputQueues output_queues;
};

}  // namespace basalt
