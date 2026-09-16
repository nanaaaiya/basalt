// One-off diagnostic: loads a calib-cam_detected_corners.cereal cache
// (written by basalt_calibrate's detect_corners step) and reports, per
// camera, how many frames were detected, the corner-count distribution,
// and the range/validity of decoded tag corner IDs -- without re-running
// the slow detection step. Built to root-cause why basalt_calibrate found
// zero frames clearing MIN_CORNERS=15 on two consecutive OAK-D Lite
// AprilGrid recordings.
//
// Usage: ./basalt_inspect_calib_corners <path/to/calib-cam_detected_corners.cereal>

#include <array>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

#include <cereal/archives/binary.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>

// Eigen<->cereal serialize() overloads -- needed to deserialize
// CalibCornerData's Eigen::Vector2d corners.
#include <basalt/serialization/headers_serialization.h>

#include <basalt/calibration/calibration_helper.h>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <cereal_path> [--export-csv cam_id out.csv]"
              << std::endl;
    return 1;
  }

  if (argc >= 5 && std::string(argv[2]) == "--export-csv") {
    int want_cam = std::stoi(argv[3]);
    std::ofstream out(argv[4]);
    out << "t_ns,corner_id,tag_id,corner_in_tag,x,y\n";
    basalt::CalibCornerMap calib_corners_export;
    basalt::CalibCornerMap calib_corners_rejected_export;
    {
      std::ifstream is(argv[1], std::ios::binary);
      cereal::BinaryInputArchive archive(is);
      archive(calib_corners_export);
      archive(calib_corners_rejected_export);
    }
    for (const auto& kv : calib_corners_export) {
      if ((int)kv.first.cam_id != want_cam) continue;
      for (size_t i = 0; i < kv.second.corner_ids.size(); i++) {
        int id = kv.second.corner_ids[i];
        out << kv.first.frame_id << "," << id << "," << (id >> 2) << ","
            << (id & 3) << "," << kv.second.corners[i](0) << ","
            << kv.second.corners[i](1) << "\n";
      }
    }
    std::cout << "Exported cam" << want_cam << " corners to " << argv[4]
              << std::endl;
    return 0;
  }

  basalt::CalibCornerMap calib_corners;
  basalt::CalibCornerMap calib_corners_rejected;

  {
    std::ifstream is(argv[1], std::ios::binary);
    if (!is) {
      std::cerr << "Can't open " << argv[1] << std::endl;
      return 1;
    }
    cereal::BinaryInputArchive archive(is);
    archive(calib_corners);
    archive(calib_corners_rejected);
  }

  std::map<int, int> frames_per_cam;
  std::map<int, long> corners_sum_per_cam;
  std::map<int, int> corners_min_per_cam, corners_max_per_cam;
  int min_id = 1 << 30, max_id = -(1 << 30);
  long total_corners = 0;
  int frames_with_zero_corners = 0;

  for (const auto& kv : calib_corners) {
    int cam = (int)kv.first.cam_id;
    int n = (int)kv.second.corners.size();
    frames_per_cam[cam]++;
    corners_sum_per_cam[cam] += n;
    total_corners += n;
    if (n == 0) frames_with_zero_corners++;

    if (corners_min_per_cam.find(cam) == corners_min_per_cam.end() ||
        n < corners_min_per_cam[cam])
      corners_min_per_cam[cam] = n;
    if (corners_max_per_cam.find(cam) == corners_max_per_cam.end() ||
        n > corners_max_per_cam[cam])
      corners_max_per_cam[cam] = n;

    for (int id : kv.second.corner_ids) {
      if (id < min_id) min_id = id;
      if (id > max_id) max_id = id;
    }
  }

  // Check simultaneous-visibility overlap: for each timestamp, do both cam0
  // and cam1 independently clear MIN_CORNERS(15)? This is what the
  // cam_graph extrinsics bootstrap actually requires -- distinct from each
  // camera's own per-frame validity.
  {
    std::map<int64_t, std::array<int, 2>> per_ts;
    for (const auto& kv : calib_corners) {
      int cam = (int)kv.first.cam_id;
      if (cam < 0 || cam > 1) continue;
      per_ts[kv.first.frame_id][cam] = (int)kv.second.corners.size();
    }
    int both_ge15 = 0, both_ge4 = 0, cam0_only = 0, cam1_only = 0, neither = 0;
    for (const auto& kv : per_ts) {
      bool c0 = kv.second[0] >= 15;
      bool c1 = kv.second[1] >= 15;
      if (c0 && c1) both_ge15++;
      if (kv.second[0] >= 4 && kv.second[1] >= 4) both_ge4++;
      if (c0 && !c1) cam0_only++;
      if (c1 && !c0) cam1_only++;
      if (!c0 && !c1) neither++;
    }
    std::cout << "timestamps total: " << per_ts.size() << std::endl;
    std::cout << "both cams >=15 corners: " << both_ge15 << std::endl;
    std::cout << "both cams >=4 corners: " << both_ge4 << std::endl;
    std::cout << "cam0>=15 only: " << cam0_only << "  cam1>=15 only: " << cam1_only
              << "  neither: " << neither << std::endl;
  }

  std::cout << "calib_corners entries (frame,cam pairs): " << calib_corners.size()
            << std::endl;
  std::cout << "calib_corners_rejected entries: " << calib_corners_rejected.size()
            << std::endl;
  std::cout << "frames with zero corners detected: " << frames_with_zero_corners
            << std::endl;
  std::cout << "corner id range across all detections: [" << min_id << ", "
            << max_id << "]  (valid range for a 6x6 grid is [0, 143])"
            << std::endl;
  std::cout << std::endl;

  for (const auto& kv : frames_per_cam) {
    int cam = kv.first;
    int frames = kv.second;
    double mean = frames > 0 ? (double)corners_sum_per_cam[cam] / frames : 0.0;
    std::cout << "cam" << cam << ": " << frames << " frames, "
              << "corners/frame min=" << corners_min_per_cam[cam]
              << " max=" << corners_max_per_cam[cam] << " mean=" << mean
              << std::endl;
  }

  // Sample a handful of individual frame entries in full detail.
  std::cout << "\n--- sample of first 10 entries ---" << std::endl;
  int shown = 0;
  for (const auto& kv : calib_corners) {
    if (shown++ >= 10) break;
    std::cout << "t_ns=" << kv.first.frame_id << " cam=" << kv.first.cam_id
              << " corners=" << kv.second.corners.size();
    if (!kv.second.corner_ids.empty()) {
      std::cout << " sample_ids=[";
      for (size_t i = 0; i < std::min<size_t>(5, kv.second.corner_ids.size());
           i++) {
        if (i) std::cout << ",";
        std::cout << kv.second.corner_ids[i];
      }
      std::cout << "]";
    }
    std::cout << std::endl;
  }

  // Dump full id + pixel-position list for the frame with the most corners,
  // per camera, so tag-ID spatial adjacency can be checked against the
  // aprilgrid.cpp assumption (tag_id = tagCols*y + x, row-major).
  std::map<int, basalt::CalibCornerData const*> best_per_cam;
  std::map<int, int> best_count_per_cam;
  for (const auto& kv : calib_corners) {
    int cam = (int)kv.first.cam_id;
    int n = (int)kv.second.corners.size();
    if (best_count_per_cam.find(cam) == best_count_per_cam.end() ||
        n > best_count_per_cam[cam]) {
      best_count_per_cam[cam] = n;
      best_per_cam[cam] = &kv.second;
    }
  }
  for (const auto& kv : best_per_cam) {
    std::cout << "\n--- full dump, cam" << kv.first << ", "
              << best_count_per_cam[kv.first] << " corners ---" << std::endl;
    const auto& d = *kv.second;
    for (size_t i = 0; i < d.corner_ids.size(); i++) {
      int id = d.corner_ids[i];
      int tag_id = id >> 2;
      int corner_in_tag = id & 3;
      std::cout << "id=" << id << " tag=" << tag_id
                << " corner=" << corner_in_tag << " x=" << d.corners[i](0)
                << " y=" << d.corners[i](1) << std::endl;
    }
  }

  return 0;
}
