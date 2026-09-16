// One-off diagnostic: loads a calib-cam_init_poses.cereal cache and reports
// num_inliers stats per camera, to check whether the PnP pose-fitting step
// is silently rejecting most correspondences as outliers despite healthy
// raw corner-detection counts.
//
// Usage: ./basalt_inspect_init_poses <init_poses.cereal> <detected_corners.cereal>

#include <array>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

#include <cereal/archives/binary.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>

#include <basalt/serialization/headers_serialization.h>

#include <basalt/calibration/calibration_helper.h>

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <init_poses.cereal> <detected_corners.cereal>" << std::endl;
    return 1;
  }

  basalt::CalibInitPoseMap init_poses;
  {
    std::ifstream is(argv[1], std::ios::binary);
    if (!is) {
      std::cerr << "Can't open " << argv[1] << std::endl;
      return 1;
    }
    cereal::BinaryInputArchive archive(is);
    archive(init_poses);
  }

  basalt::CalibCornerMap calib_corners;
  basalt::CalibCornerMap calib_corners_rejected;
  {
    std::ifstream is(argv[2], std::ios::binary);
    if (!is) {
      std::cerr << "Can't open " << argv[2] << std::endl;
      return 1;
    }
    cereal::BinaryInputArchive archive(is);
    archive(calib_corners);
    archive(calib_corners_rejected);
  }

  std::map<int, long> sum_inliers, sum_corners, count;
  std::map<int, int> min_inliers, max_inliers;

  for (const auto& kv : init_poses) {
    int cam = (int)kv.first.cam_id;
    size_t inliers = kv.second.num_inliers;
    sum_inliers[cam] += inliers;
    count[cam]++;
    if (min_inliers.find(cam) == min_inliers.end() || (int)inliers < min_inliers[cam])
      min_inliers[cam] = (int)inliers;
    if (max_inliers.find(cam) == max_inliers.end() || (int)inliers > max_inliers[cam])
      max_inliers[cam] = (int)inliers;

    auto cit = calib_corners.find(kv.first);
    if (cit != calib_corners.end()) sum_corners[cam] += cit->second.corners.size();
  }

  std::cout << "init_poses entries: " << init_poses.size() << std::endl;
  for (const auto& kv : count) {
    int cam = kv.first;
    std::cout << "cam" << cam << ": " << kv.second << " entries, "
              << "num_inliers mean=" << (double)sum_inliers[cam] / kv.second
              << " min=" << min_inliers[cam] << " max=" << max_inliers[cam]
              << " | mean raw corners=" << (double)sum_corners[cam] / kv.second
              << std::endl;
  }

  // Distribution of num_inliers (histogram buckets) per camera.
  std::map<int, std::map<int, int>> hist;  // cam -> bucket -> count
  for (const auto& kv : init_poses) {
    int cam = (int)kv.first.cam_id;
    int inliers = (int)kv.second.num_inliers;
    int bucket = inliers / 5 * 5;
    hist[cam][bucket]++;
  }
  for (const auto& kv : hist) {
    std::cout << "\nnum_inliers histogram, cam" << kv.first << ":" << std::endl;
    for (const auto& b : kv.second) {
      std::cout << "  [" << b.first << "-" << b.first + 4 << "]: " << b.second
                << std::endl;
    }
  }

  // Sample: show a handful of frames where raw corners were high but
  // num_inliers came out low, to see the drop directly.
  std::cout << "\n--- sample: raw_corners vs num_inliers (first 20 per cam) ---"
            << std::endl;
  std::map<int, int> shown;
  for (const auto& kv : init_poses) {
    int cam = (int)kv.first.cam_id;
    if (shown[cam]++ >= 20) continue;
    auto cit = calib_corners.find(kv.first);
    size_t raw = cit != calib_corners.end() ? cit->second.corners.size() : (size_t)-1;
    std::cout << "cam" << cam << " t=" << kv.first.frame_id
              << " raw_corners=" << raw << " num_inliers=" << kv.second.num_inliers
              << std::endl;
  }

  return 0;
}
