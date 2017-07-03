#ifndef EYE_CENTER_ASCEND_PAUL_H
#define EYE_CENTER_ASCEND_PAUL_H

#include <opencv2/opencv.hpp>

using namespace cv;

namespace EyeCenterAscendPaul {
  int findEyeCenters(Mat& image, Point*& centers, bool silentMode);
}

#endif
