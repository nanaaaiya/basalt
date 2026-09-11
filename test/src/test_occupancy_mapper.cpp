#include <basalt/mapping/occupancy_mapper.h>

#include <chrono>
#include <thread>

#include "gtest/gtest.h"

namespace {

// A single-camera calibration with identity extrinsics (camera frame ==
// IMU/world frame) and a simple pinhole model -- deliberately no
// distortion, so every math step in insertFrame() is hand-checkable.
basalt::Calibration<double> makeTestCalib() {
  basalt::Calibration<double> calib;
  calib.T_i_c.push_back(Sophus::SE3d());  // identity

  basalt::GenericCamera<double> cam;
  // fx=fy=100, principal point at (50, 50) -- matches the 100x100 test
  // depth images below exactly, so the center pixel's bearing vector is
  // exactly (0, 0, 1) with no rounding to worry about.
  cam.variant = basalt::PinholeCamera<double>(Eigen::Vector4d(100, 100, 50, 50));
  calib.intrinsics.push_back(cam);

  return calib;
}

// depth_mm at the exact center pixel (the principal point), 0 everywhere
// else -- isolates the test to one, exactly-predictable ray.
cv::Mat makeCenterPixelDepth(uint16_t depth_mm) {
  cv::Mat depth = cv::Mat::zeros(100, 100, CV_16UC1);
  depth.at<uint16_t>(50, 50) = depth_mm;
  return depth;
}

// Records that the calibration's own intrinsics were computed for a
// 100x100 image (matching makeTestCalib()'s fx=fy=100, cx=cy=50), for the
// resolution-mismatch test below -- makeTestCalib() itself leaves
// calib.resolution empty, which every other test above relies on to keep
// insertFrame()'s scale factor at a no-op 1.0.
basalt::Calibration<double> makeTestCalibWithResolution(int w, int h) {
  basalt::Calibration<double> calib = makeTestCalib();
  calib.resolution.push_back(Eigen::Vector2i(w, h));
  return calib;
}

// pollVoxelDelta() is fed by an async processing thread -- poll with a
// generous timeout rather than assuming it's ready the instant
// addDepthFrame() returns.
bool pollWithTimeout(basalt::OccupancyMapper& mapper, basalt::VoxelDelta& out,
                      int timeout_ms = 2000) {
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    if (mapper.pollVoxelDelta(out)) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return false;
}

}  // namespace

TEST(OccupancyMapperTest, SingleRayMarksOneOccupiedVoxel) {
  basalt::OccupancyMapper mapper(makeTestCalib(), /*voxel_size=*/0.2,
                                  /*depth_stride=*/1);
  mapper.start();

  auto frame = std::make_shared<basalt::DepthFrameInput>();
  frame->t_ns = 0;
  frame->T_w_c = Sophus::SE3d();  // camera at world origin, identity rotation
  frame->depth_mm = makeCenterPixelDepth(2100);  // 2.1m -- deliberately not
  // an exact multiple of the 0.2m voxel size, to avoid floating-point
  // voxel-boundary ambiguity in this test.
  frame->cam_id = 0;
  mapper.addDepthFrame(frame);

  basalt::VoxelDelta delta;
  ASSERT_TRUE(pollWithTimeout(mapper, delta));
  ASSERT_EQ(delta.added.size(), 1u);
  EXPECT_TRUE(delta.removed.empty());

  // The occupied voxel's center should be within one voxel of (0,0,2) --
  // the center pixel's bearing is exactly (0,0,1), so the back-projected
  // point is exactly (0,0,2), and octomap snaps it to its containing
  // voxel's center.
  const Eigen::Vector3d& p = delta.added[0];
  EXPECT_NEAR(p.x(), 0.0, 0.2);
  EXPECT_NEAR(p.y(), 0.0, 0.2);
  EXPECT_NEAR(p.z(), 2.1, 0.2);

  mapper.stop();
}

TEST(OccupancyMapperTest, ReinsertingTheSamePointProducesNoChange) {
  basalt::OccupancyMapper mapper(makeTestCalib(), /*voxel_size=*/0.2,
                                  /*depth_stride=*/1);
  mapper.start();

  auto frame1 = std::make_shared<basalt::DepthFrameInput>();
  frame1->T_w_c = Sophus::SE3d();
  frame1->depth_mm = makeCenterPixelDepth(2100);
  mapper.addDepthFrame(frame1);

  basalt::VoxelDelta delta1;
  ASSERT_TRUE(pollWithTimeout(mapper, delta1));
  ASSERT_EQ(delta1.added.size(), 1u);

  // Same observation again -- the voxel is already occupied, so this
  // should report nothing (pollVoxelDelta only ever surfaces a *change*,
  // not a full resend), directly exercising the same octomap
  // change-detection behavior proven in isolation by the Step A smoke
  // test, now through this class's own code path.
  auto frame2 = std::make_shared<basalt::DepthFrameInput>();
  frame2->T_w_c = Sophus::SE3d();
  frame2->depth_mm = makeCenterPixelDepth(2100);
  mapper.addDepthFrame(frame2);

  basalt::VoxelDelta delta2;
  bool got_second = pollWithTimeout(mapper, delta2, /*timeout_ms=*/500);
  if (got_second) {
    EXPECT_TRUE(delta2.added.empty());
    EXPECT_TRUE(delta2.removed.empty());
  }
  // If nothing was even pushed to the output queue (the more likely
  // outcome, since insertFrame() only try_pushes a non-empty delta),
  // that's equally correct -- there's simply nothing to report.

  mapper.stop();
}

TEST(OccupancyMapperTest, RayPastAnOccupiedVoxelClearsIt) {
  basalt::OccupancyMapper mapper(makeTestCalib(), /*voxel_size=*/0.2,
                                  /*depth_stride=*/1);
  mapper.start();

  // First: something at 2m looks occupied.
  auto frame1 = std::make_shared<basalt::DepthFrameInput>();
  frame1->T_w_c = Sophus::SE3d();
  frame1->depth_mm = makeCenterPixelDepth(2100);
  mapper.addDepthFrame(frame1);

  basalt::VoxelDelta delta1;
  ASSERT_TRUE(pollWithTimeout(mapper, delta1));
  ASSERT_EQ(delta1.added.size(), 1u);

  // Second: real returns now come from 4.1m along the exact same ray --
  // proves free-space ray-casting works through this class, not just
  // occupancy marking. octomap accumulates evidence as log-odds rather
  // than overwriting on the latest observation, and its defaults are
  // deliberately asymmetric: one hit (+0.847 log-odds) immediately clears
  // the occupied threshold (0), but one miss only subtracts ~0.405 --
  // leaving a once-hit cell still (barely) occupied after a single
  // contradicting observation. That's intentional, real behavior (a
  // single noisy miss shouldn't erase solid prior evidence), not a bug --
  // it takes ceil(0.847/0.405) = 3 consecutive misses to actually flip
  // it, so this sends 3 to observe the real transition rather than
  // asserting on the first one.
  Eigen::aligned_vector<Eigen::Vector3d> all_added, all_removed;
  for (int i = 0; i < 3; i++) {
    auto frame = std::make_shared<basalt::DepthFrameInput>();
    frame->T_w_c = Sophus::SE3d();
    frame->depth_mm = makeCenterPixelDepth(4100);
    mapper.addDepthFrame(frame);

    basalt::VoxelDelta delta;
    if (pollWithTimeout(mapper, delta, /*timeout_ms=*/500)) {
      all_added.insert(all_added.end(), delta.added.begin(), delta.added.end());
      all_removed.insert(all_removed.end(), delta.removed.begin(),
                          delta.removed.end());
    }
  }

  ASSERT_EQ(all_added.size(), 1u);   // the new 4.1m voxel, seen on the first hit
  ASSERT_EQ(all_removed.size(), 1u);  // the old 2.1m voxel, cleared by the 3rd miss
  EXPECT_NEAR(all_added[0].z(), 4.1, 0.2);
  EXPECT_NEAR(all_removed[0].z(), 2.1, 0.2);

  mapper.stop();
}

TEST(OccupancyMapperTest, AllInvalidDepthProducesNoDelta) {
  basalt::OccupancyMapper mapper(makeTestCalib(), /*voxel_size=*/0.2);
  mapper.start();

  auto frame = std::make_shared<basalt::DepthFrameInput>();
  frame->T_w_c = Sophus::SE3d();
  frame->depth_mm = cv::Mat::zeros(100, 100, CV_16UC1);  // all invalid (0)
  mapper.addDepthFrame(frame);

  basalt::VoxelDelta delta;
  EXPECT_FALSE(pollWithTimeout(mapper, delta, /*timeout_ms=*/300));

  mapper.stop();
}

TEST(OccupancyMapperTest, DepthResolutionLowerThanCalibrationIsScaled) {
  // Reproduces the real "fan-shaped map" bug: the depth stream outputs at
  // half the calibration's resolution (mirrors DepthAI's StereoDepth
  // DEFAULT preset producing 320x240 depth against a 640x480 calibration).
  // The calibration here is declared for a 100x100 image (fx=fy=100,
  // cx=cy=50), but the depth frame is only 50x50 -- so the true principal
  // point (50, 50) lands at (25, 25) in the actual depth image. Without
  // the fix, pixel (25, 25) is unprojected directly, landing far off the
  // optical axis and back-projecting to the wrong point; with the fix, it
  // scales up to (50, 50) before unprojecting, giving exactly the same
  // on-axis (0, 0, depth) result as SingleRayMarksOneOccupiedVoxel above.
  basalt::OccupancyMapper mapper(makeTestCalibWithResolution(100, 100),
                                  /*voxel_size=*/0.2, /*depth_stride=*/1);
  mapper.start();

  auto frame = std::make_shared<basalt::DepthFrameInput>();
  frame->T_w_c = Sophus::SE3d();
  frame->depth_mm = cv::Mat::zeros(50, 50, CV_16UC1);
  frame->depth_mm.at<uint16_t>(25, 25) = 2100;  // half-res principal point
  mapper.addDepthFrame(frame);

  basalt::VoxelDelta delta;
  ASSERT_TRUE(pollWithTimeout(mapper, delta));
  ASSERT_EQ(delta.added.size(), 1u);

  const Eigen::Vector3d& p = delta.added[0];
  EXPECT_NEAR(p.x(), 0.0, 0.2);
  EXPECT_NEAR(p.y(), 0.0, 0.2);
  EXPECT_NEAR(p.z(), 2.1, 0.2);

  mapper.stop();
}

TEST(OccupancyMapperTest, StopIsSafeWithoutStart) {
  // stop() must be safe to call even if start() never ran (mirrors
  // OakDDevice/DashboardClient's own guard idiom) -- also covers the
  // destructor path, since ~OccupancyMapper() calls stop().
  basalt::OccupancyMapper mapper(makeTestCalib(), /*voxel_size=*/0.2);
  mapper.stop();
}
