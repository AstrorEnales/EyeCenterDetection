#ifndef EYE_CENTER_EVOLALG_H
#define EYE_CENTER_EVOLALG_H

#include <opencv2/opencv.hpp>

using namespace cv;

namespace EyeCenterEvolAlg {
  int findEyeCenters(Mat& image, Point*& centers, bool silentMode);
}

#endif
