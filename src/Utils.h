#ifndef EYE_CENTER_UTILS_H
#define EYE_CENTER_UTILS_H

#include <opencv2/opencv.hpp>

using namespace cv;

void showNormalizedImage(const Mat& img, std::string name);

bool bordersReached(Point c, int w, int h);

double fitness(Mat& image, Mat& grad_x, Mat& grad_y, Point c);

#endif
