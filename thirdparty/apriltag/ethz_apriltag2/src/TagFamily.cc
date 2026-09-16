#include <iostream>

#include "apriltags/TagFamily.h"

/**

// example of instantiation of tag family:

#include "apriltags/TagFamily.h"
#include "apriltags/Tag36h11.h"
TagFamily *tag36h11 = new TagFamily(tagCodes36h11);

// available tag families:

#include "apriltags/Tag16h5.h"
#include "apriltags/Tag16h5_other.h"
#include "apriltags/Tag25h7.h"
#include "apriltags/Tag25h9.h"
#include "apriltags/Tag36h11.h"
#include "apriltags/Tag36h11_other.h"
#include "apriltags/Tag36h9.h"

*/


namespace AprilTags {

TagFamily::TagFamily(const TagCodes& tagCodes, const size_t blackBorder)
  : blackBorder(blackBorder), bits(tagCodes.bits), dimension((int)std::sqrt((float)bits)),
    minimumHammingDistance(tagCodes.minHammingDistance),
    // Was hardcoded to 1 -- unusually conservative for tag36h11
    // (minimumHammingDistance=11, which safely supports correcting up to
    // (11-1)/2=5 bit errors: any two valid codes differ by >=11 bits, so a
    // 5-bit correction can never accidentally match the wrong codeword).
    // Confirmed via a real capture (basalt calibration on OAK-D Lite
    // AprilGrid photos) that the old default of 1 was rejecting genuine,
    // correctly-printed tag detections with observed hamming distances of
    // 3-5 as "not good enough" purely due to this threshold -- not a
    // sensor, printing, or lighting problem. thisTagFamily is const on
    // TagDetector, so this has to be set here at construction rather than
    // via setErrorRecoveryBits()/setErrorRecoveryFraction() afterward.
    //
    // Deliberately NOT the theoretical max (5): raising it that far let
    // enough marginal/borderline reads through that calibration kept
    // converging to a bad-but-stable local minimum (~31px mean
    // reprojection error) regardless of optimizer strategy -- consistent
    // with some accepted detections being genuinely mis-decoded (a wrong
    // codeword that happens to land within the correction budget of the
    // true one), corrupting the 2D<->3D correspondence for those points.
    // 3 and 4 were both tried: 3 is too strict (zero valid frames again,
    // same failure as the old broken default of 1); 4 gave very little
    // data (5-6 frames) but the SAME ~30px reprojection bias as 5's 196
    // frames -- that data-volume-independence pointed at a systematic
    // intrinsics-seeding bug (see cam_calib.cpp's xi/alpha fix), not a
    // wrong-decode-rate problem after all. Back to the max (5) now that
    // the actual cause is fixed, since it gave the most usable data.
    errorRecoveryBits(5), codes() {
  if ( bits != dimension*dimension )
    cerr << "Error: TagFamily constructor called with bits=" << bits << "; must be a square number!" << endl;
  codes = tagCodes.codes;
}

void TagFamily::setErrorRecoveryBits(int b) {
  errorRecoveryBits = b;
}

void TagFamily::setErrorRecoveryFraction(float v) {
  errorRecoveryBits = (int) (((int) (minimumHammingDistance-1)/2)*v);
}

unsigned long long TagFamily::rotate90(unsigned long long w, int d) {
  unsigned long long wr = 0;
  const unsigned long long oneLongLong = 1;

  for (int r = d-1; r>=0; r--) {
    for (int c = 0; c<d; c++) {
      int b = r + d*c;
      wr = wr<<1;
      
      if ((w & (oneLongLong<<b)) != 0)
	wr |= 1;
    }
  }
  return wr;
}

int TagFamily::hammingDistance(unsigned long long a, unsigned long long b) {
  return popCount(a^b);
}

unsigned char TagFamily::popCountReal(unsigned long long w) {
  unsigned char cnt = 0;
  while (w != 0) {
    w &= (w-1);
    ++cnt;
  }
  return cnt;
}

int TagFamily::popCount(unsigned long long w) {
  int count = 0;
  while (w != 0) {
    count += popCountTable[(unsigned int) (w & (popCountTableSize-1))];
    w >>= popCountTableShift;
  }
  return count;
}

void TagFamily::decode(TagDetection& det, unsigned long long rCode) const {
  int  bestId = -1;
  int  bestHamming = INT_MAX;
  int  bestRotation = 0;
  unsigned long long bestCode = 0;

  unsigned long long rCodes[4];
  rCodes[0] = rCode;
  rCodes[1] = rotate90(rCodes[0], dimension);
  rCodes[2] = rotate90(rCodes[1], dimension);
  rCodes[3] = rotate90(rCodes[2], dimension);

  for (unsigned int id = 0; id < codes.size(); id++) {
    for (unsigned int rot = 0; rot < 4; rot++) {
      int thisHamming = hammingDistance(rCodes[rot], codes[id]);
      if (thisHamming < bestHamming) {
	bestHamming = thisHamming;
	bestRotation = rot;
	bestId = id;
	bestCode = codes[id];
      }
    }
  }
  det.id = bestId;
  det.hammingDistance = bestHamming;
  det.rotation = bestRotation;
  det.good = (det.hammingDistance <= errorRecoveryBits);
  det.obsCode = rCode;
  det.code = bestCode;
}

void TagFamily::printHammingDistances() const {
  vector<int> hammings(dimension*dimension+1);
  for (unsigned i = 0; i < codes.size(); i++) {
    unsigned long long r0 = codes[i];
    unsigned long long r1 = rotate90(r0, dimension);
    unsigned long long r2 = rotate90(r1, dimension);
    unsigned long long r3 = rotate90(r2, dimension);
    for (unsigned int j = i+1; j < codes.size(); j++) {
      int d = min(min(hammingDistance(r0, codes[j]),
		      hammingDistance(r1, codes[j])),
		  min(hammingDistance(r2, codes[j]),
		      hammingDistance(r3, codes[j])));
      hammings[d]++;
    }
  }

  for (unsigned int i = 0; i < hammings.size(); i++)
    printf("hammings: %u = %d\n", i, hammings[i]);
}

unsigned char TagFamily::popCountTable[TagFamily::popCountTableSize];

TagFamily::TableInitializer TagFamily::initializer;

} // namespace
