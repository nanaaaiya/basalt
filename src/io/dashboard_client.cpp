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

#include <basalt/io/dashboard_client.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

#include <nlohmann/json.hpp>

namespace basalt {

namespace {

// Minimal base64 encoder -- no new dependency needed (nlohmann::json
// doesn't ship one, and this is short enough not to be worth pulling in a
// library for). Matches Python's base64.b64encode() output exactly,
// including padding.
std::string base64Encode(const unsigned char* data, size_t len) {
  static const char* kTable =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve(((len + 2) / 3) * 4);

  size_t i = 0;
  while (i + 3 <= len) {
    uint32_t n = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2];
    out += kTable[(n >> 18) & 0x3F];
    out += kTable[(n >> 12) & 0x3F];
    out += kTable[(n >> 6) & 0x3F];
    out += kTable[n & 0x3F];
    i += 3;
  }
  size_t rem = len - i;
  if (rem == 1) {
    uint32_t n = data[i] << 16;
    out += kTable[(n >> 18) & 0x3F];
    out += kTable[(n >> 12) & 0x3F];
    out += "==";
  } else if (rem == 2) {
    uint32_t n = (data[i] << 16) | (data[i + 1] << 8);
    out += kTable[(n >> 18) & 0x3F];
    out += kTable[(n >> 12) & 0x3F];
    out += kTable[(n >> 6) & 0x3F];
    out += "=";
  }
  return out;
}

// Reconnect backoff: fast enough that a brief hiccup barely gets missed,
// capped low enough that a genuinely down backend (or Wi-Fi drop) doesn't
// spin the CPU or spam connect() -- 1s doubling to 8s.
constexpr int kBackoffMinMs = 1000;
constexpr int kBackoffMaxMs = 8000;

}  // namespace

DashboardClient::DashboardClient(std::string host, int port)
    : host_(std::move(host)), port_(port) {
  out_queue_.set_capacity(200);
  incoming_save_map_queue_.set_capacity(4);
}

DashboardClient::~DashboardClient() { stop(); }

void DashboardClient::start() {
  running_ = true;
  connection_thread_ = std::thread([this] { connectionThreadMain(); });
}

void DashboardClient::stop() {
  running_ = false;
  if (connection_thread_.joinable()) connection_thread_.join();
}

void DashboardClient::connectionThreadMain() {
  int backoff_ms = kBackoffMinMs;

  while (running_) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
      std::cerr << "[DashboardClient] socket() failed: " << std::strerror(errno)
                << std::endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
      backoff_ms = std::min(backoff_ms * 2, kBackoffMaxMs);
      continue;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port_));
    if (::inet_pton(AF_INET, host_.c_str(), &addr.sin_addr) != 1) {
      std::cerr << "[DashboardClient] invalid host: " << host_ << std::endl;
      ::close(fd);
      return;  // not a transient error -- misconfiguration, don't retry forever
    }

    // TCP_NODELAY: pose messages are small and latency-sensitive (they
    // drive the live HUD/trajectory); Nagle's algorithm would add up to
    // ~40ms of pointless buffering per message on a link this quiet.
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
      ::close(fd);
      std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
      backoff_ms = std::min(backoff_ms * 2, kBackoffMaxMs);
      continue;
    }

    std::cout << "[DashboardClient] connected to " << host_ << ":" << port_
              << std::endl;
    connected_ = true;
    backoff_ms = kBackoffMinMs;

    std::thread reader([this, fd] { readerThreadMain(fd); });
    writerThreadMain(fd);  // blocks here until the connection drops or stop()

    // readerThreadMain is parked in a blocking recv() that only notices
    // running_ going false on its next wakeup -- which, on a healthy
    // connection where the backend never hangs up, is never. shutdown()
    // (not close() -- fd must stay valid for the join below, and get
    // closed exactly once) forces that recv() to return immediately so
    // stop()/the destructor can't deadlock waiting for this join.
    ::shutdown(fd, SHUT_RDWR);
    reader.join();

    connected_ = false;
    ::close(fd);
    std::cout << "[DashboardClient] disconnected, will retry" << std::endl;
  }
}

void DashboardClient::writerThreadMain(int fd) {
  while (running_) {
    std::string line;
    // 200ms poll instead of an unbounded pop() so this thread notices
    // running_ going false (stop()) or the reader thread's fd having died
    // promptly instead of blocking indefinitely on an empty queue.
    if (!out_queue_.try_pop(line)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      continue;
    }
    line += '\n';
    size_t sent = 0;
    while (sent < line.size()) {
      ssize_t n = ::send(fd, line.data() + sent, line.size() - sent, MSG_NOSIGNAL);
      if (n <= 0) return;  // connection dead -- let connectionThreadMain reconnect
      sent += static_cast<size_t>(n);
    }
  }
}

void DashboardClient::readerThreadMain(int fd) {
  std::string buf;
  char chunk[4096];
  while (running_) {
    ssize_t n = ::recv(fd, chunk, sizeof(chunk), 0);
    if (n <= 0) return;  // connection dead
    buf.append(chunk, static_cast<size_t>(n));

    size_t pos;
    while ((pos = buf.find('\n')) != std::string::npos) {
      std::string line = buf.substr(0, pos);
      buf.erase(0, pos + 1);

      try {
        auto j = nlohmann::json::parse(line);
        if (j.value("type", "") == "command" &&
            j.value("cmd", "") == "save_map") {
          incoming_save_map_queue_.try_push(j.value("run_id", ""));
        }
      } catch (const std::exception& e) {
        std::cerr << "[DashboardClient] bad JSON from backend, skipping: "
                  << e.what() << std::endl;
      }
    }
  }
}

bool DashboardClient::pollSaveMapCommand(std::string& run_id_out) {
  return incoming_save_map_queue_.try_pop(run_id_out);
}

void DashboardClient::sendPose(int64_t t_ns, bool corrected,
                                const Eigen::Vector3d& p,
                                const Eigen::Vector4d& quat_xyzw,
                                const Eigen::Vector3d* vel_w_i) {
  nlohmann::json j;
  j["type"] = "pose";
  j["run_id"] = "ignored";  // backend overwrites this per-connection
  j["source"] = "pi5";
  j["t_ns"] = t_ns;
  j["frame"] = corrected ? "corrected" : "raw";
  j["position"] = {p.x(), p.y(), p.z()};
  j["orientation"] = {quat_xyzw.x(), quat_xyzw.y(), quat_xyzw.z(), quat_xyzw.w()};
  if (vel_w_i) {
    j["velocity"] = {vel_w_i->x(), vel_w_i->y(), vel_w_i->z()};
  }
  out_queue_.try_push(j.dump());
}

void DashboardClient::sendMapEvent(int64_t t_ns, const std::string& event,
                                    const std::string& detail_json) {
  nlohmann::json j;
  j["type"] = "map_event";
  j["run_id"] = "ignored";
  j["event"] = event;
  j["t_ns"] = t_ns;
  try {
    j["detail"] = nlohmann::json::parse(detail_json);
  } catch (const std::exception&) {
    j["detail"] = nlohmann::json::object();
  }
  out_queue_.try_push(j.dump());
}

bool DashboardClient::sendMapFile(const std::string& name,
                                   const std::string& ply_path) {
  std::ifstream f(ply_path, std::ios::binary);
  if (!f) {
    std::cerr << "[DashboardClient] can't open " << ply_path << std::endl;
    return false;
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  std::string bytes = ss.str();

  nlohmann::json j;
  j["type"] = "map_file";
  j["run_id"] = "ignored";
  j["name"] = name;
  j["ply_b64"] = base64Encode(
      reinterpret_cast<const unsigned char*>(bytes.data()), bytes.size());

  // This one deliberately bypasses out_queue_ (which is sized for many
  // small pose/map_event messages, not one multi-MB payload) and blocks
  // briefly waiting for room -- see the class comment on why that's fine
  // here specifically.
  std::string line = j.dump();
  for (int attempt = 0; attempt < 50; ++attempt) {  // ~5s worth of retries
    if (out_queue_.try_push(line)) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  std::cerr << "[DashboardClient] sendMapFile: queue stayed full, giving up"
            << std::endl;
  return false;
}

bool writePointCloudPly(const std::string& path,
                         const Eigen::aligned_vector<Eigen::Vector3d>& points) {
  std::ofstream os(path);
  if (!os) return false;
  os << "ply\nformat ascii 1.0\nelement vertex " << points.size()
     << "\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
  for (const auto& p : points) os << p.x() << " " << p.y() << " " << p.z() << "\n";
  return true;
}

}  // namespace basalt
