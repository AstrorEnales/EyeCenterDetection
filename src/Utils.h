#ifndef EYE_CENTER_UTILS_H
#define EYE_CENTER_UTILS_H

#include <opencv2/opencv.hpp>

using namespace cv;

void showNormalizedImage(const Mat& img, std::string name) {
  Mat imgNormalized;
  normalize(img, imgNormalized, 1, 0, NORM_MINMAX);
  imshow(name, imgNormalized);
}

#endif
