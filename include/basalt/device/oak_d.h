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
  // Auto-exposure always uses the longest allowed exposure before raising
  // gain, so the cap directly sets motion blur (~ angular rate x exposure x
  // focal length). At 8 ms, fast turns (150-300 deg/s) smeared features by
  // 9-18 px and tracking collapsed; indoors at 4 ms the camera raises gain
  // (ISO ~650 -> ~1300 of ~1550 max) and keeps the same brightness with no
  // measurable extra noise, halving blur. Raise it (--max-exposure-us) in
  // dim rooms where gain maxes out and images turn dark.
  static constexpr int DEFAULT_MAX_EXPOSURE_US = 4000;

  // enable_stereo_depth: builds and runs the on-device StereoDepth node
  // (for depth recording) alongside the existing raw
  // mono + IMU streams VIO uses. This has to be a constructor-time
  // choice, not a setter called after start() like setOutputQueues()
  // below -- DepthAI's node graph is fixed once pipeline.start() runs,
  // so whether the StereoDepth node exists at all can't be decided
  // later. Defaults to false so every existing caller (just
  // `new OakDDevice`) keeps paying zero extra device-side compute for a
  // stream it never asked for.
  // depth_full_res: feed StereoDepth from separate native 1280x800 mono
  // outputs at DEPTH_FULL_RES_FPS (VIO keeps its 640x480 30 fps streams).
  // ~1.7x finer depth than the 640x480 crop with the full FOV; 10 fps keeps
  // the device from saturating (at 30 fps VIO's own streams fell to 17 fps).
  explicit OakDDevice(bool enable_stereo_depth = false, bool depth_full_res = true,
                      int max_exposure_us = DEFAULT_MAX_EXPOSURE_US);
  ~OakDDevice();

  void start();
  void stop();

  void setOutputQueues(
      tbb::concurrent_bounded_queue<OpticalFlowInput::Ptr>* image_queue,
      tbb::concurrent_bounded_queue<ImuData<double>::Ptr>* imu_queue);
  void detachOutputQueues();

  // Separate from setOutputQueues() above -- depth is for recording, an
  // entirely separate consumer from VIO's
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

  // Plain pinhole intrinsics for whatever camera StereoDepth's depth
  // output is rectified/aligned to (currently CAM_B/left -- no explicit
  // setDepthAlign() call exists, so this assumes DepthAI's default;
  // see start()'s comment). "Plain pinhole" is deliberate, not an
  // approximation: StereoDepth's own depth output is ALREADY undistorted
  // by construction (block-matching depth algorithms fundamentally
  // require rectified input), so no distortion coefficients apply here at
  // all -- unlike calib_.intrinsics[cam_id], which describes the RAW,
  // un-rectified lens (needed for VIO/optical-flow, which reads raw
  // frames, but wrong for depth). fx/fy/cx/cy are reported at
  // `width`x`height`.
  struct DepthIntrinsics {
    double fx = 0, fy = 0, cx = 0, cy = 0;
    int width = 0, height = 0;
  };

  // Only meaningful once start() has run (queries the live device's own
  // factory calibration) -- see start()'s comment for why this can't be
  // known any earlier. Live-diagnosed (2026-09-29, OAK-D Pro W): using
  // calib_.intrinsics[cam_id] (Basalt's own kb4 fit of the RAW lens) to
  // unproject StereoDepth's RECTIFIED output produced a severely warped
  // point cloud -- Luxonis's own factory calibration (queried here) has a
  // 21% different focal length AND a completely different distortion
  // model family (their own "Perspective"/8-parameter model vs our kb4
  // fit), because StereoDepth rectifies internally using ITS OWN
  // calibration, never Basalt's. Harmless/unused on the OAK-D Lite, where
  // the two calibrations happened to be close enough that this mismatch
  // was too small to notice.
  DepthIntrinsics getDepthIntrinsics() const { return depth_intrinsics_; }

  // Exposure of the most recent left (cam0) frame in microseconds, -1 before
  // the first frame. Read from frame metadata, not a camera query.
  int lastExposureUs() const { return last_exposure_us_; }

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
  const bool depth_full_res_;
  const int max_exposure_us_;
  std::atomic<int> last_exposure_us_{-1};
  static constexpr float DEPTH_FULL_RES_FPS = 10.0f;
  DepthIntrinsics depth_intrinsics_;

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
