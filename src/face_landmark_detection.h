#ifndef FACE_LANDMARK_DETECTION_H
#define FACE_LANDMARK_DETECTION_H

#include <opencv2/opencv.hpp>

using namespace cv;

namespace FD {
  void getEyes(Mat& image, String classifierPath, bool setNull, Mat& leftEye, Mat& rightEye,
               Point& leftEyeOffset, Point& rightEyeOffset);
}

#endif
