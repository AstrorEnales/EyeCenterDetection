#include "Utils.h"
#include "Gradient.h"
#include "EyeCenterNaive.h"
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

int EyeCenterNaive::findEyeCenters(Mat& image, Point*& centers) {
  //GaussianBlur(image, image, Size(5, 5), 0, 0, BORDER_DEFAULT);

  Mat grey(image.size(), CV_8UC1);
  cvtColor(image, grey, CV_RGB2GRAY);

  Mat fitnessImage(image.size(), CV_32FC1);

  Mat grad_x, grad_y;
  calculateGradients(Four_Neighbor, grey, grad_x, grad_y);
    
  /*
  //set gradient border to 0
  for(int x = 0; x < grey.cols; x++) {
    grad_y.at<float>(0, x) = 0;
    grad_x.at<float>(0, x) = 0;
    grad_y.at<float>(grey.rows-1, x) = 0;
    grad_x.at<float>(grey.rows-1, x) = 0;
  }
  for(int y = 0; y < grey.rows; y++) {
    grad_y.at<float>(y, 0) = 0;
    grad_x.at<float>(y, 0) = 0;
    grad_y.at<float>(y, grey.cols-1) = 0;
    grad_x.at<float>(y, grey.cols-1) = 0;
  }
  */

  int N = image.cols * image.rows;
  float percentage_step = 100.0 / N;

  double fitness = 0;
  double fitness_factor = 1.0 / N;
  double length, dot;
  Point2f d, g;

  for(int y = 0; y < grad_x.rows; y++) {
    for(int x = 0; x < grad_x.cols; x++) {
      float gx = grad_x.at<float>(y, x);
      float gy = grad_y.at<float>(y, x);
      length = sqrt(gx * gx + gy * gy);
      if(length > 0) {
        length = 1 / length;
        grad_x.at<float>(y, x) = gx * length;
        grad_y.at<float>(y, x) = gy * length;
      }
    }
  }

  for(int y = 0; y < image.rows; y++) {
    std::cout << (percentage_step * (y * image.cols)) << "%" << std::endl;
    for(int x = 0; x < image.cols; x++) {
      fitness = 0;
      for(int y2 = 0; y2 < image.rows; y2++) {
        for(int x2 = 0; x2 < image.cols; x2++) {
          // Normalized distance vector
          d = Point2f(x2 - x, y2 - y);
          length = sqrt(d.x * d.x + d.y * d.y);
          if(length > 0) {
            d *= 1 / length;
          }
          // Normalized gradient vector
          g = Point2f(grad_x.at<float>(y2, x2), grad_y.at<float>(y2, x2));
          dot = d.dot(g);
          fitness += dot * dot;
        }
      }
      fitness = fitness * fitness_factor;
      fitnessImage.at<float>(y, x) = fitness;
    }
  }

  // Set fitness border 0
  for(int y = 0; y < grad_x.rows; y++) {
    fitnessImage.at<float>(y, 0) = 0;
    fitnessImage.at<float>(y, fitnessImage.cols - 1) = 0;
  }
  for(int x = 0; x < grad_x.cols; x++) {
    fitnessImage.at<float>(0, x) = 0;
    fitnessImage.at<float>(fitnessImage.rows - 1, x) = 0;
  }

  showNormalizedImage(grad_x, "gradx");
  showNormalizedImage(grad_y, "grady");
  showNormalizedImage(fitnessImage, "fit");

  // Find max fitness location
  double min, max;
  Point min_loc, max_loc;
  minMaxLoc(fitnessImage, &min, &max, &min_loc, &max_loc);
  int target_x = max_loc.x;
  int target_y = max_loc.y;

  centers = new Point[1];
  centers[0] = max_loc;
  return 1;
}

