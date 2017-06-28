#ifndef EYE_CENTER_UTILS_H
#define EYE_CENTER_UTILS_H

#include <opencv2/opencv.hpp>

using namespace cv;

void showNormalizedImage(const Mat& img, std::string name);

inline bool bordersReached(const int cx, const int cy, const int w, const int h) {
  return cx <= 0 || cx >= w - 1 || cy <= 0 || cy >= h - 1;
}

void buildDisplacementLookup(Mat& displacementLookup, int w, int h);

double fitness(const int cols, const int rows, Mat& grad_x, Mat& grad_y, Mat& displacementLookup, int x, int y);

#endif
