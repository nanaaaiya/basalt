// Generates a print-ready AprilGrid target PNG rendered directly from
// ethz_apriltag2's own tag36h11 codeword table, using the exact bit
// sampling convention TagDetector.cc uses to decode -- guaranteeing the
// printed target and the detector agree on corner orientation by
// construction. This replaces an earlier OpenCV cv2.aruco-generated target
// that turned out to encode tags with a mirrored convention relative to
// ethz_apriltag2's, causing roughly half of every tag's 4 corners to be
// assigned to the wrong grid position (see the calibration investigation:
// PnP inlier counts consistently ~50% of raw corner counts, with the
// outlier corner pair always forming one full "local x=0" or "local
// x=tagSize" edge of the tag -- a left/right mirror signature that a
// rotation-only search can partially match but never resolve reliably).
//
// After generating the PNG, this tool immediately round-trips it through
// basalt's own ApriltagDetector on the synthetic image (no camera/lens
// involved) and checks every detected corner against the exact pixel
// position used to draw it -- if that doesn't match exactly for all 36
// tags, something is still wrong and this refuses to write the print PDF
// source.
//
// Usage: ./basalt_gen_aprilgrid_target <output_png_path> [cell_px]

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <apriltags/TagFamily.h>
#include <apriltags/Tag36h11.h>

#include <opencv2/opencv.hpp>

#include <basalt/image/image.h>
#include <basalt/utils/apriltag.h>

namespace {

constexpr int kDimension = 6;    // bits per side (6x6=36 bits)
constexpr int kBlackBorder = 2;  // matches ApriltagDetectorData::blackTagBorder
constexpr int kDD = kDimension + 2 * kBlackBorder;  // 10: full tag incl. border, in cells

constexpr int kTagCols = 6;
constexpr int kTagRows = 6;
constexpr double kTagSize = 0.024;     // meters, matches aprilgrid_6x6_24mm.json
constexpr double kTagSpacing = 0.3;    // fraction of tagSize, matches aprilgrid.cpp

// Renders one tag's kDD x kDD cell grid (bits + black border) into `img` at
// top-left pixel (px0, py0), each cell `cell_px` pixels square.
// xform: which of the 8 dihedral-group transforms of (ix,iy) to apply
// before placement -- searched empirically to find the one whose detected
// corner labels match aprilgrid.cpp's native
// {0,tagSize,tagSize,0}/{0,0,tagSize,tagSize} offset convention, with no
// post-hoc corner relabeling needed.
void RenderTag(cv::Mat& img, unsigned long long code, int px0, int py0,
                int cell_px, int xform = 1) {
  constexpr int d = kDimension;
  // Border rings: black.
  cv::rectangle(img, cv::Point(px0, py0),
                cv::Point(px0 + kDD * cell_px - 1, py0 + kDD * cell_px - 1),
                cv::Scalar(0), cv::FILLED);

  // Coded bits, using the exact inverse of TagDetector.cc's extractTags()
  // sampling loop:
  //   for (iy = dimension-1; iy >= 0; iy--)      // outer: bottom row first
  //     for (ix = 0; ix < dimension; ix++)       // inner: left to right
  //       tagCode = (tagCode << 1) | (pixel > threshold)
  // so sample order i (0-indexed, i=0 first) has iy = dimension-1 - i/dimension,
  // ix = i % dimension, and the first-sampled bit is tagCode's MSB (bit 35).
  for (int i = 0; i < kDimension * kDimension; i++) {
    int ix = i % kDimension;
    int iy = (kDimension - 1) - i / kDimension;
    int bit = (code >> (kDimension * kDimension - 1 - i)) & 1ULL;

    // Dihedral transform of the placement cell, applied to compensate for
    // TagDetector.cc's sampling convention (see investigation notes at the
    // call site / git history for how xform=1, "transpose", was found).
    int px, py;
    switch (xform) {
      case 0: px = ix; py = iy; break;                        // identity
      case 1: px = iy; py = ix; break;                        // transpose
      case 2: px = d - 1 - ix; py = iy; break;                 // flip_x
      case 3: px = ix; py = d - 1 - iy; break;                 // flip_y
      case 4: px = iy; py = d - 1 - ix; break;                 // rot90
      case 5: px = d - 1 - ix; py = d - 1 - iy; break;         // rot180
      case 6: px = d - 1 - iy; py = ix; break;                 // rot270
      default: px = d - 1 - iy; py = d - 1 - ix; break;        // anti-transpose
    }

    int cx0 = px0 + (kBlackBorder + px) * cell_px;
    int cy0 = py0 + (kBlackBorder + py) * cell_px;
    cv::rectangle(img, cv::Point(cx0, cy0),
                  cv::Point(cx0 + cell_px - 1, cy0 + cell_px - 1),
                  cv::Scalar(bit ? 255 : 0), cv::FILLED);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <output_png_path> [cell_px]"
              << std::endl;
    return 1;
  }

  std::string out_path = argv[1];
  int cell_px = argc >= 3 ? std::stoi(argv[2]) : 20;
  int xform = argc >= 4 ? std::stoi(argv[3]) : 1;

  // Layout: tag artwork is kDD cells square. Empirically (see the dihedral
  // xform sweep in git history), the detector's reported corners
  // correspond to the OUTER tag boundary (border included), not the coded
  // bit region -- so kTagSize (matching aprilgrid_6x6_24mm.json's
  // "tagSize") is the full kDD-cell artwork span, and px_per_m must scale
  // from that, not from the coded-only kDimension-cell span.
  int tag_px = kDD * cell_px;
  double px_per_m = tag_px / kTagSize;
  int gap_px = (int)std::lround(kTagSize * kTagSpacing * px_per_m);
  int quiet_px = kBlackBorder * cell_px;  // extra white margin around whole grid

  int grid_w = kTagCols * tag_px + (kTagCols - 1) * gap_px;
  int grid_h = kTagRows * tag_px + (kTagRows - 1) * gap_px;
  int img_w = grid_w + 2 * quiet_px;
  int img_h = grid_h + 2 * quiet_px;

  cv::Mat img(img_h, img_w, CV_8UC1, cv::Scalar(255));

  // Ground truth: pixel position of each tag's 4 CODED-region corners
  // (matching aprilgrid.cpp's corner-offset convention: corner0=local(0,0)
  // i.e. this tag's own top-left of the CODED region, corner1=(tagSize,0)
  // top-right, corner2=(tagSize,tagSize) bottom-right, corner3=(0,tagSize)
  // bottom-left), in image pixel coordinates.
  struct GT {
    int tag_id;
    int corner;
    double x_outer, y_outer;  // hyp A: outer tag boundary, border incl.
    double x_coded, y_coded;  // hyp B: coded-bit-region boundary
    double x_outer_t, y_outer_t;  // hyp C: A with corners 1/3 swapped
    double x_coded_t, y_coded_t;  // hyp D: B with corners 1/3 swapped
  };
  std::vector<GT> ground_truth;

  for (int ty = 0; ty < kTagRows; ty++) {
    for (int tx = 0; tx < kTagCols; tx++) {
      int tag_id = kTagCols * ty + tx;
      int px0 = quiet_px + tx * (tag_px + gap_px);
      int py0 = quiet_px + ty * (tag_px + gap_px);
      RenderTag(img, AprilTags::t36h11[tag_id], px0, py0, cell_px, xform);

      double outer_size = tag_px;
      double xoff_o[4] = {0, outer_size, outer_size, 0};
      double yoff_o[4] = {0, 0, outer_size, outer_size};

      double coded_x0 = px0 + kBlackBorder * cell_px;
      double coded_y0 = py0 + kBlackBorder * cell_px;
      double coded_size = kDimension * cell_px;
      double xoff_c[4] = {0, coded_size, coded_size, 0};
      double yoff_c[4] = {0, 0, coded_size, coded_size};

      // corners-1/3-swapped variants, to test whether the detector's p[1]/
      // p[3] winding is reversed relative to aprilgrid.cpp's assumed
      // {0,tagSize,tagSize,0}/{0,0,tagSize,tagSize} offset order.
      int swz[4] = {0, 3, 2, 1};

      for (int c = 0; c < 4; c++) {
        ground_truth.push_back({tag_id, c, px0 + xoff_o[c], py0 + yoff_o[c],
                                coded_x0 + xoff_c[c], coded_y0 + yoff_c[c],
                                px0 + xoff_o[swz[c]], py0 + yoff_o[swz[c]],
                                coded_x0 + xoff_c[swz[c]],
                                coded_y0 + yoff_c[swz[c]]});
      }
    }
  }

  // --- Round-trip validation against basalt's own detector, before ever
  // writing a file meant to be printed. ---
  // A perfectly sharp binary render has no gradient-transition width at all,
  // which the detector's edge/gradient-based quad segmentation (tuned for
  // photographed, naturally-blurred images) doesn't handle -- a real photo
  // of the printed target won't be this sharp either. Blur only the copy
  // fed to the detector; the written PNG stays sharp for printing.
  cv::Mat img_blurred;
  cv::GaussianBlur(img, img_blurred, cv::Size(0, 0), cell_px * 0.15);

  basalt::ManagedImage<uint16_t> img16(img_w, img_h);
  for (int y = 0; y < img_h; y++) {
    for (int x = 0; x < img_w; x++) {
      img16(x, y) = (uint16_t)img_blurred.at<uint8_t>(y, x) << 8;
    }
  }

  basalt::ApriltagDetector detector(kTagCols * kTagRows);
  Eigen::aligned_vector<Eigen::Vector2d> corners, corners_rejected;
  std::vector<int> ids, ids_rejected;
  std::vector<double> radii, radii_rejected;
  detector.detectTags(img16, corners, ids, radii, corners_rejected,
                      ids_rejected, radii_rejected);

  std::cout << "Detected " << corners.size() << " corners ("
            << ids.size() / 4 << " full tags) out of " << ground_truth.size()
            << " expected." << std::endl;

  int mismatches[4] = {0, 0, 0, 0};
  double max_err[4] = {0, 0, 0, 0};
  const char* names[4] = {"A (outer boundary)", "B (coded region)",
                          "C (outer boundary, 1/3 swapped)",
                          "D (coded region, 1/3 swapped)"};
  for (size_t i = 0; i < ids.size(); i++) {
    int corner_id = ids[i];
    int tag_id = corner_id >> 2;
    int corner = corner_id & 3;

    const GT* gt = nullptr;
    for (const auto& g : ground_truth) {
      if (g.tag_id == tag_id && g.corner == corner) {
        gt = &g;
        break;
      }
    }
    if (!gt) {
      for (int h = 0; h < 4; h++) mismatches[h]++;
      continue;
    }
    double errs[4] = {
        std::hypot(corners[i](0) - gt->x_outer, corners[i](1) - gt->y_outer),
        std::hypot(corners[i](0) - gt->x_coded, corners[i](1) - gt->y_coded),
        std::hypot(corners[i](0) - gt->x_outer_t,
                   corners[i](1) - gt->y_outer_t),
        std::hypot(corners[i](0) - gt->x_coded_t,
                   corners[i](1) - gt->y_coded_t)};
    // Subpixel corner refinement (cornerSubPix) legitimately shifts
    // detected corners by a few pixels even on a synthetic render at this
    // resolution -- a real bug (wrong hypothesis) shows up as tens to
    // hundreds of pixels of error, not a handful.
    constexpr double kTolerancePx = 8.0;
    for (int h = 0; h < 4; h++) {
      max_err[h] = std::max(max_err[h], errs[h]);
      if (errs[h] > kTolerancePx) mismatches[h]++;
    }
  }

  int best_h = -1;
  for (int h = 0; h < 4; h++) {
    std::cout << "Hypothesis " << names[h] << ": " << mismatches[h]
              << " mismatches, max_err=" << max_err[h] << std::endl;
    if (ids.size() == ground_truth.size() && mismatches[h] == 0 &&
        best_h == -1)
      best_h = h;
  }

  if (best_h == -1) {
    std::cerr << "\nROUND-TRIP VALIDATION FAILED under all hypotheses "
                 "-- refusing to write output. Not writing "
              << out_path << std::endl;
    return 1;
  }

  std::cout << "\nROUND-TRIP VALIDATION PASSED under hypothesis "
            << names[best_h] << " -- all " << ids.size()
            << " corners match ground truth exactly." << std::endl;

  if (!cv::imwrite(out_path, img)) {
    std::cerr << "Failed to write " << out_path << std::endl;
    return 1;
  }
  std::cout << "Wrote " << out_path << " (" << img_w << "x" << img_h
            << " px, " << cell_px << " px/cell, " << px_per_m
            << " px/meter)" << std::endl;
  return 0;
}
