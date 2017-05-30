#ifndef GRADIENT_H
#define GRADIENT_H

#include <opencv2/opencv.hpp>

enum GradientType {
  OpenCV_Scharr,
  Four_Neighbor
};

void calculateGradients(GradientType type, cv::Mat& grey, cv::Mat& grad_x, cv::Mat& grad_y);

#endif
