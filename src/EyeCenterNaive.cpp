#include "Utils.h"
#include "Gradient.h"
#include "EyeCenterNaive.h"
#include <stdio.h>

using namespace cv;

int EyeCenterNaive::findEyeCenters(Mat& image, Point*& centers, bool silentMode) {
  //GaussianBlur(image, image, Size(5, 5), 0, 0, BORDER_DEFAULT);

  Mat grey(image.size(), CV_8UC1);
  cvtColor(image, grey, CV_RGB2GRAY);

  Mat fitnessImage(image.size(), CV_32FC1, float(0));

  Mat grad_x, grad_y;
  calculateGradients(Four_Neighbor, grey, grad_x, grad_y);

  Mat displacementLookup(Size(image.cols * 2, image.rows * 2), CV_32FC2);
  Point displacementLookupCenter(image.cols, image.rows);
  float length;
  for(int y = 0; y < displacementLookup.rows; y++) {
    for(int x = 0; x < displacementLookup.cols; x++) {
      // Normalized distance vector
      float dy = y - displacementLookupCenter.y;
      float dx = x - displacementLookupCenter.x;
      length = sqrt(dx * dx + dy * dy);

      if(length > 0) {
        dx /= length;
        dy /= length;
      }
      displacementLookup.at<Point2f>(y, x) = Point2f(dx, dy);
    }
  }

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
  float percentage_step = 100.0f / N;

  int dx, dy;
  float gx, gy, dot;
  Point2f d;

  for(int y = 0; y < grad_x.rows; y++) {
    for(int x = 0; x < grad_x.cols; x++) {
      gx = grad_x.at<float>(y, x);
      gy = grad_y.at<float>(y, x);
      length = sqrt(gx * gx + gy * gy);
      if(length > 0) {
        length = 1 / length;
        grad_x.at<float>(y, x) = gx * length;
        grad_y.at<float>(y, x) = gy * length;
      }
    }
  }
  
  for(int y2 = 0; y2 < image.rows; y2++) {
    if (!silentMode) std::cout << (percentage_step * (y2 * image.cols)) << "%" << std::endl;
    float* grad_ptr_x = grad_x.ptr<float>(y2);
    float* grad_ptr_y = grad_y.ptr<float>(y2);
    for(int x2 = 0; x2 < image.cols; x2++) {
      gx = grad_ptr_x[x2];
      gy = grad_ptr_y[x2];
      dx = displacementLookupCenter.x + x2;
      dy = displacementLookupCenter.y + y2;
      for(int y = 0; y < image.rows; y++) {
        float* fitness_ptr = fitnessImage.ptr<float>(y);
        Point2f* displacement_lookup_ptr = displacementLookup.ptr<Point2f>(dy - y);
        for(int x = 0; x < image.cols; x++) {
          d = displacement_lookup_ptr[dx - x];
          dot = max(0.0f, d.x * gx + d.y * gy);
          fitness_ptr[x] += dot * dot;
        }
      }
    }
  }
  fitnessImage /= N;

  // Set fitness border 0
  for(int y = 0; y < grad_x.rows; y++) {
    fitnessImage.at<float>(y, 0) = 0;
    fitnessImage.at<float>(y, fitnessImage.cols - 1) = 0;
  }
  for(int x = 0; x < grad_x.cols; x++) {
    fitnessImage.at<float>(0, x) = 0;
    fitnessImage.at<float>(fitnessImage.rows - 1, x) = 0;
  }

  if (!silentMode) {
    showNormalizedImage(grad_x, "gradx");
    showNormalizedImage(grad_y, "grady");
    showNormalizedImage(fitnessImage, "fit");
  }

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
