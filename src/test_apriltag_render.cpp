// Tests basalt's ApriltagDetector against a real captured photo, after
// raising errorRecoveryBits from its old default of 1 to the theoretical
// safe max for tag36h11 (5), to check whether that alone explains the
// zero-detections failure.

#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <basalt/image/image.h>
#include <basalt/utils/apriltag.h>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <path_to_png>" << std::endl;
    return 1;
  }

  cv::Mat mat = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
  if (mat.empty()) {
    std::cerr << "Can't read " << argv[1] << std::endl;
    return 1;
  }

  basalt::ManagedImage<uint16_t> img(mat.cols, mat.rows);
  for (int y = 0; y < mat.rows; y++)
    for (int x = 0; x < mat.cols; x++)
      img(x, y) = ((uint16_t)mat.at<uint8_t>(y, x)) << 8;

  basalt::ApriltagDetector ad(36);
  Eigen::aligned_vector<Eigen::Vector2d> corners, corners_rej;
  std::vector<int> ids, ids_rej;
  std::vector<double> radii, radii_rej;

  ad.detectTags(img, corners, ids, radii, corners_rej, ids_rej, radii_rej);

  std::cout << argv[1] << ": found " << ids.size() << " tag(s), ids = [";
  for (size_t i = 0; i < ids.size(); i++) {
    if (i) std::cout << ",";
    std::cout << ids[i];
  }
  std::cout << "]" << std::endl;

  if (argc >= 3 && std::string(argv[2]) == "--dump-corners") {
    for (size_t i = 0; i < ids.size(); i++) {
      int id = ids[i];
      std::cout << "CORNER id=" << id << " tag=" << (id >> 2)
                << " corner=" << (id & 3) << " x=" << corners[i](0)
                << " y=" << corners[i](1) << std::endl;
    }
  }

  return 0;
}
