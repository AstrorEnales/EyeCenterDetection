#ifndef GRADIENT_H
#define GRADIENT_H

#include <opencv2/opencv.hpp>

using namespace cv;

enum GradientType {
  OpenCV_Scharr,
  Four_Neighbor
};

void calculateGradients(GradientType type, Mat& grey, Mat& grad_x, Mat& grad_y);

void normalizeGradients(Mat& grad_x, Mat& grad_y);

#endif
