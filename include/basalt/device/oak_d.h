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

// Live driver for a Luxonis OAK-D Lite (stereo mono cameras + BMI270 IMU),
// modeled directly on RsT265Device's public shape (start()/stop()/
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

  OpticalFlowInput::Ptr getLastImageData() const;

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

  struct OutputQueues {
    tbb::concurrent_bounded_queue<OpticalFlowInput::Ptr>* image_data_queue =
        nullptr;
    tbb::concurrent_bounded_queue<ImuData<double>::Ptr>* imu_data_queue =
        nullptr;
    tbb::concurrent_bounded_queue<std::shared_ptr<dai::ImgFrame>>*
        depth_data_queue = nullptr;
  };

  mutable std::mutex output_queues_mutex;
  OutputQueues output_queues;
};

}  // namespace basalt
