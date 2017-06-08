#ifndef EYE_CENTER_NAIVE_H
#define EYE_CENTER_NAIVE_H

#include <opencv2/opencv.hpp>

using namespace cv;

namespace EyeCenterNaive {
  int findEyeCenters(Mat& image, Point*& centers, bool silentMode);
}

#endif
