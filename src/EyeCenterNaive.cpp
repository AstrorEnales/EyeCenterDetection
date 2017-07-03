#include "Utils.h"
#include "Gradient.h"
#include "EyeCenterNaive.h"
#include <stdio.h>

using namespace cv;

int EyeCenterNaive::findEyeCenters(Mat& image, Point*& centers, bool silentMode) {
  Mat grey(image.size(), CV_8UC1);
  cvtColor(image, grey, CV_RGB2GRAY);

  Mat fitnessImage(image.size(), CV_32FC1, float(0));

  Mat grad_x, grad_y;
  calculateGradients(Four_Neighbor, grey, grad_x, grad_y);
  
  Mat displacementLookup;
  buildDisplacementLookup(displacementLookup, image.cols, image.rows);

  int N = image.cols * image.rows;
  float percentage_step = 100.0f / N;

  int dx, dy;
  float gx, gy, dlx, dly, dot;
  Point displacementLookupCenter(image.cols, image.rows);
  Point2f d;

  normalizeGradients(grad_x, grad_y);
  
  int y2, x2, y, x, i;
  for(y2 = 0; y2 < image.rows; y2++) {
    if (!silentMode) std::cout << (percentage_step * (y2 * image.cols)) << "%" << std::endl;
    const float* grad_ptr_x = grad_x.ptr<float>(y2);
    const float* grad_ptr_y = grad_y.ptr<float>(y2);
    for(x2 = 0; x2 < image.cols; x2++) {
      gx = grad_ptr_x[x2];
      gy = grad_ptr_y[x2];
      dx = displacementLookupCenter.x + x2;
      dy = displacementLookupCenter.y + y2;
      for(y = 0; y < image.rows; y++) {
        float* fitness_ptr = fitnessImage.ptr<float>(y);
        const float* displacement_lookup_ptr = displacementLookup.ptr<float>(dy - y);
        i = dx * 2;
        for(x = 0; x < image.cols; x++) {
          dlx = displacement_lookup_ptr[i--];
          dly = displacement_lookup_ptr[i--];
          dot = dlx * gx + dly * gy;
          //if(dot > 0)
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
