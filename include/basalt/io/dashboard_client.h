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

// Sends live VIO state to the VIO Dashboard backend and receives commands
// back from it (currently just "save_map"), over a single TCP connection
// carrying newline-delimited JSON -- see VIO_Dashboard/backend/app/schema.py
// for the authoritative message shapes this must match field-for-field.
//
// This class is the CLIENT: it dials out to the dashboard backend, not the
// other way around. Two reasons, both carried over from how the dashboard's
// Pi5 ingestion side was designed: (1) the backend never needs to know this
// device's IP, which matters once there's more than one companion computer
// on the fleet; (2) this device (Wi-Fi, USB peripherals, battery) is the
// less reliable half of the link, so it should be the one that notices a
// drop and reconnects, not the more stable laptop side.
//
// Threading model, matching the try_push/drop-if-full idiom already used
// elsewhere in this codebase (e.g. OnlineLoopClosure::localization_queue,
// vio_plot_queue in oak_d_vio.cpp): sendPose/sendImage/sendMapEvent are
// called from hot VIO/GUI threads and must never block on network I/O, so
// they only ever push onto a bounded queue and return immediately, silently
// dropping the message if the queue is full (the connection is down, or
// badly lagging) rather than backing up the caller. A dedicated writer
// thread owns the actual socket writes. sendMapFile() is the one exception
// -- it's called once, right after a map is built, from a thread that has
// nothing else to do while it waits, so it blocks until sent (or times out).
//
// NOT YET RUN AGAINST A REAL DASHBOARD BACKEND: compiles and links cleanly
// (verified against this project's real vcpkg headers and the
// basalt_oak_d_vio target), but do a real test (host+port pointed at a
// running `uvicorn app.main:app`, INGESTION_SOURCE=pi5) before trusting it
// on a real flight.

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include <tbb/concurrent_queue.h>

#include <Eigen/Core>
#include <opencv2/core/mat.hpp>

#include <basalt/utils/eigen_utils.hpp>

namespace basalt {

class DashboardClient {
 public:
  using Ptr = std::shared_ptr<DashboardClient>;

  // host/port: the laptop backend's address and PI5_TCP_PORT (default
  // 8765). run_id is NOT sent by this client -- the backend assigns one
  // per TCP connection and overwrites whatever this device puts in that
  // field, so it's fine to leave it as a placeholder in every outgoing
  // message; see pi5_adapter.py's _handle_client().
  DashboardClient(std::string host, int port);
  ~DashboardClient();

  DashboardClient(const DashboardClient&) = delete;
  DashboardClient& operator=(const DashboardClient&) = delete;

  // Starts the background connect/reconnect + reader + writer threads.
  void start();
  // Stops all threads and closes the socket. Safe to call even if the
  // connection is currently down.
  void stop();

  // Non-blocking, best-effort -- see threading model above.
  void sendPose(int64_t t_ns, bool corrected, const Eigen::Vector3d& p,
                const Eigen::Vector4d& quat_xyzw,
                const Eigen::Vector3d* vel_w_i = nullptr);

  // frame: BGR or grayscale, whatever oak_d.cpp already hands the GUI --
  // this JPEG-encodes it internally. cam_id matches the dashboard's
  // existing cam0/cam1 convention (see cameraPanels.js).
  void sendImage(int cam_id, int64_t t_ns, const cv::Mat& frame,
                 const std::vector<cv::Point2f>& keypoints = {});

  // event: one of "loop_closure" | "keyframe" | "map_saved" | "reinit",
  // matching MapEventType in schema.py exactly (any other string is
  // forwarded as-is and the dashboard will just show it verbatim in a
  // toast -- see mapEvents.js's LABELS fallback).
  void sendMapEvent(int64_t t_ns, const std::string& event,
                     const std::string& detail_json = "{}");

  // Call once a map .ply is ready (e.g. after
  // OnlineLoopClosure::buildPointCloud() is written out), in response to a
  // pollSaveMapCommand() hit. Blocks (briefly -- this is a rare, deliberate
  // action, not a hot-path call) until the file is sent or a timeout
  // elapses. Returns false on failure (not connected, file unreadable, send
  // timed out).
  bool sendMapFile(const std::string& name, const std::string& ply_path);

  // Non-blocking poll for an incoming "save_map" command (see
  // MapCommand in schema.py). Returns true and fills run_id_out at most
  // once per command received; call this once per GUI frame the same
  // way oak_d_vio.cpp already drains vio_plot_queue/localization_queue.
  bool pollSaveMapCommand(std::string& run_id_out);

  bool isConnected() const { return connected_; }

 private:
  void connectionThreadMain();  // owns connect/reconnect + spawns reader/writer
  void readerThreadMain(int fd);
  void writerThreadMain(int fd);

  std::string host_;
  int port_;

  std::atomic<bool> running_{false};
  std::atomic<bool> connected_{false};
  std::thread connection_thread_;

  // Outgoing: bounded, drop-oldest-effectively-never (try_push just fails
  // silently when full -- see .cpp) so a lagging/dead connection can never
  // make VIO/GUI threads block.
  tbb::concurrent_bounded_queue<std::string> out_queue_;

  // Incoming save_map commands, drained by pollSaveMapCommand().
  tbb::concurrent_bounded_queue<std::string> incoming_save_map_queue_;
};

// Writes points as an ASCII PLY -- the point cloud OnlineLoopClosure::
// buildPointCloud() returns, read back and sent up via sendMapFile(). A
// free function, not a DashboardClient method, since it has nothing to do
// with the connection itself.
bool writePointCloudPly(const std::string& path,
                         const Eigen::aligned_vector<Eigen::Vector3d>& points);

}  // namespace basalt
