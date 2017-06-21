#include "Gradient.h"

using namespace cv;

float fourNeighborX(int y, int x, const Mat& grey) {
  int intensityMinus = (int) grey.at<uchar>(y, max(0, x - 1));
  int intensityPlus = (int) grey.at<uchar>(y, min(grey.cols - 1, x + 1));
  return (intensityPlus - intensityMinus) * 0.5f;
}

float fourNeighborY(int y, int x, const Mat& grey) {
  int intensityMinus = (int) grey.at<uchar>(max(0, y - 1), x);
  int intensityPlus = (int) grey.at<uchar>(min(grey.rows - 1, y + 1), x);
  return (intensityPlus - intensityMinus) * 0.5f;
}

void calculateGradients(GradientType type, Mat& grey, Mat& grad_x, Mat& grad_y) {
  switch(type) {
    case OpenCV_Scharr:
      Scharr(grey, grad_x, CV_32FC1, 1, 0);
      Scharr(grey, grad_y, CV_32FC1, 0, 1);
      break;

    case Four_Neighbor:
      grad_x = Mat(grey.size(), CV_32FC1);
      grad_y = Mat(grey.size(), CV_32FC1);
      for (int x = 0; x < grey.cols; x++) {
        for (int y = 0; y < grey.rows; y++) {
          grad_x.at<float>(y, x) = fourNeighborX(y, x, grey);
          grad_y.at<float>(y, x) = fourNeighborY(y, x, grey);
        }
      }
      break;
  }
}

