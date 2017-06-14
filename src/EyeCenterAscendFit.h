#ifndef EYE_CENTER_ASCEND_FIT_H
#define EYE_CENTER_ASCEND_FIT_H

#include <opencv2/opencv.hpp>

using namespace cv;

namespace EyeCenterAscendFit {
  int findEyeCenters(Mat& image, Point*& centers, bool silentMode);
}

#endif
