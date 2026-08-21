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

OakDDevice::OakDDevice() {}

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

  pipeline.start();
  std::cout << "[OAKD]: device connected, streaming" << std::endl;

  device_thread = std::thread(&OakDDevice::deviceLoop, this);
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

  while (running.load() && pipeline.isRunning()) {
    bool got_data = false;

    OutputQueues queues;
    {
      std::lock_guard<std::mutex> lock(output_queues_mutex);
      queues = output_queues;
    }

    while (auto imuData = q_imu->tryGet<dai::IMUData>()) {
      got_data = true;
      for (auto& packet : imuData->packets) {
        double t = to_seconds(packet.acceleroMeter.getTimestamp());

        ImuData<double>::Ptr data;
        data.reset(new ImuData<double>);
        data->t_ns = (int64_t)(t * 1e9);
        data->accel << packet.acceleroMeter.x, packet.acceleroMeter.y,
            packet.acceleroMeter.z;
        data->gyro << packet.gyroscope.x, packet.gyroscope.y,
            packet.gyroscope.z;

        if (queues.imu_data_queue) queues.imu_data_queue->push(data);
        last_imu_time = t;
      }
    }

    while (auto frame = q_left->tryGet<dai::ImgFrame>()) {
      got_data = true;
      left_queue.push_back({to_seconds(frame->getTimestamp()), frame});
    }
    while (auto frame = q_right->tryGet<dai::ImgFrame>()) {
      got_data = true;
      right_queue.push_back({to_seconds(frame->getTimestamp()), frame});
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
        for (size_t j = 0; j < full_size; j++) {
          int val = data_in[j];
          val = val << 8;
          data_out[j] = val;
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

void OakDDevice::detachOutputQueues() {
  std::lock_guard<std::mutex> lock(output_queues_mutex);
  output_queues = {};
}

OpticalFlowInput::Ptr OakDDevice::getLastImageData() const {
  std::lock_guard<std::mutex> lock(last_img_data_mutex);
  return last_img_data;
}

}  // namespace basalt
