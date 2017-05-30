#ifndef EYE_CENTER_ASCEND_H
#define EYE_CENTER_ASCEND_H

#include <opencv2/opencv.hpp>

using namespace cv;

namespace EyeCenterAscend {
  int findEyeCenters(Mat& image, Point*& centers);
}

#endif
