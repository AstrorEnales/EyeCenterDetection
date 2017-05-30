#include "Utils.h"
#include "Gradient.h"
#include "EyeCenterAscend.h"
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

int EyeCenterAscend::findEyeCenters(Mat& image, Point*& centers) {
  //GaussianBlur(image, image, Size(5, 5), 0, 0, BORDER_DEFAULT);

  Mat grey(image.size(), CV_8UC1);
  cvtColor(image, grey, CV_RGB2GRAY);

  Mat grad_x, grad_y;
  calculateGradients(Four_Neighbor, grey, grad_x, grad_y);

  int m = 100;
  int tmax = 20;
  int N = image.cols * image.rows;

  // Find the m highest magnitudes
  float highest_magnitudes[m];
  Point highest_magnitude_pixels[m];
  for(int y = 0; y < grad_x.rows; y++) {
    for(int x = 0; x < grad_x.cols; x++) {
      float gx = grad_x.at<float>(y, x);
      float gy = grad_y.at<float>(y, x);
      float mag = sqrt(gx * gx + gy * gy);
      
      int index = -1;
      while(mag > highest_magnitudes[index + 1] && index < m) {
        index += 1;
      }
      if(index > -1) {
        for(int i = 0; i < index; i++) {
          highest_magnitudes[i] = highest_magnitudes[i + 1];
          highest_magnitude_pixels[i] = highest_magnitude_pixels[i + 1];
        }
        highest_magnitudes[index] = mag;
        highest_magnitude_pixels[index] = Point(x, y);
      }
    }
  }

  for(int i = 0; i < m; i++) {
    Point c = highest_magnitude_pixels[i];
    Point c_old;
    for(int j = 0; j < tmax; j++) {
      c_old = c;
      float g = 0;
      for(int y = 0; y < image.rows; y++) {
        for(int x = 0; x < image.cols; x++) {
          // TODO
        }
      }
      g = g * 2 / N;
    }
  }

  // TODO
  centers = new Point[1];
  centers[0] = Point(0, 0);
  return 1;
}
