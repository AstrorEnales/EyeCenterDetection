#include "Utils.h"
#include "Gradient.h"
#include "EyeCenterAscendFit.h"
#include <stdio.h>
#include <math.h>  

using namespace cv;

float getFitnessLookup(const int cols, const int rows, Mat& grad_x, Mat& grad_y, Mat& fitnessLookup,
                       Mat& displacementLookup, int x, int y) {
  float& val = fitnessLookup.at<float>(y, x);
  if(val < 0)
    val = fitness(cols, rows, grad_x, grad_y, displacementLookup, x, y);
  return val;
}

int EyeCenterAscendFit::findEyeCenters(Mat& image, Point*& centers, bool silentMode) {
  Mat grey(image.size(), CV_8UC1);
  cvtColor(image, grey, CV_RGB2GRAY);

  Mat debugImage;
  if(!silentMode) debugImage = image.clone();

  Mat fitnessLookup(image.size(), CV_32FC1, float(-1));
  
  Mat grad_x, grad_y;
  calculateGradients(Four_Neighbor, grey, grad_x, grad_y);
  
  const int m = 10;
  const int tmax = 40;
  const int N = image.cols * image.rows;
  const double sigma = 0.001;
  const int stepSizeCount = 10;
  const double stepIntervalMin = pow(10, -2);
  const double stepIntervalMax = pow(10, 1);

  double interval;
  double stepSizes[stepSizeCount];
  for (int h = 0; h < stepSizeCount; h++) {
    interval = (exp(h) - 1) / (exp(stepSizeCount - 1) - 1);
    interval = (interval * (stepIntervalMax - stepIntervalMin) + stepIntervalMin);
    stepSizes[h] = interval;
  }

  float maxMagnitude = -1, minMagnitude = -1;
  
  // Find the m highest magnitudes
  float highest_magnitudes[m];
  for(int i = 0; i < m; i++) {
    highest_magnitudes[i] = -1;
  }
  int highest_magnitude_pixels_x[m];
  int highest_magnitude_pixels_y[m];
  int x, i, index;
  float gx, gy, length;
  for(int y = 0; y < grad_x.rows; y++) {
    const float* grad_x_ptr = grad_x.ptr<float>(y);
    const float* grad_y_ptr = grad_y.ptr<float>(y);
    for(x = 0; x < grad_x.cols; x++) {
      gx = grad_x_ptr[x];
      gy = grad_y_ptr[x];
      length = gx * gx + gy * gy;
      index = -1;
      while(length > highest_magnitudes[index + 1] && index < m - 1) {
        index += 1;
      }
      if(index > -1) {
        for(i = 0; i < index; i++) {
          highest_magnitudes[i] = highest_magnitudes[i + 1];
          highest_magnitude_pixels_x[i] = highest_magnitude_pixels_x[i + 1];
          highest_magnitude_pixels_y[i] = highest_magnitude_pixels_y[i + 1];
        }
        highest_magnitudes[index] = length;
        highest_magnitude_pixels_x[index] = x;
        highest_magnitude_pixels_y[index] = y;
      }   
    }
  }
  
  normalizeGradients(grad_x, grad_y);
  
  Mat displacementLookup;
  buildDisplacementLookup(displacementLookup, image.cols, image.rows);
  Point displacementLookupCenter(image.cols, image.rows);
  
  // Set fitness border 0
  for(int y = 0; y < image.rows; y++) {
    fitnessLookup.at<float>(y, 0) = 0;
    fitnessLookup.at<float>(y, image.cols - 1) = 0;
  }
  for(int x = 0; x < image.cols; x++) {
    fitnessLookup.at<float>(0, x) = 0;
    fitnessLookup.at<float>(image.rows - 1, x) = 0;
  }

  Point centerPoints[m];
  double centerFitness[m];
  int j, h, cx, cy, coldX, coldY, ctestX, ctestY;
  float left, right, top, bottom;
  double bestFitness, bestStepSize, f;
  Point d_i, g_i;
  for(int i = 0; i < m; i++) {
    cx = highest_magnitude_pixels_x[i];
    cy = highest_magnitude_pixels_y[i];
    for(j = 0; j < tmax; j++) {
      coldX = cx;
      coldY = cy;
      
      // Fitness border is set to 0, so early out in that case
      left = cx == 0 ? 0 : getFitnessLookup(image.cols, image.rows, grad_x, grad_y, fitnessLookup, displacementLookup, cx - 1, cy);
      top = cy == 0 ? 0 : getFitnessLookup(image.cols, image.rows, grad_x, grad_y, fitnessLookup, displacementLookup, cx, cy - 1);
      right = cx == image.cols - 1 ? 0 : getFitnessLookup(image.cols, image.rows, grad_x, grad_y, fitnessLookup, displacementLookup, cx + 1, cy);
      bottom = cy == image.rows - 1 ? 0 : getFitnessLookup(image.cols, image.rows, grad_x, grad_y, fitnessLookup, displacementLookup, cx, cy + 1);
      //* edgy movement
      gx = right == left ? 0 : right < left ? -1 : 1;
      gy = bottom == top ? 0 : bottom < top ? -1 : 1;
      //*/
      /* Fluid movement
      gx = (right - left) * 0.5f;
      gy = (bottom - top) * 0.5f;
      length = sqrt(gx * gx + gy * gy);
      if (length > 0) {
        gx /= length;
        gy /= length;
      }
      //*/
      bestFitness = -1;
      bestStepSize = 0;
      for (h = 0; h < stepSizeCount; h++) {
        interval = stepSizes[h];
        ctestX = (int)(cx + interval * gx);
        ctestY = (int)(cy + interval * gy);
        if (!bordersReached(ctestX, ctestY, image.cols, image.rows)) {
          f = fitness(image.cols, image.rows, grad_x, grad_y, displacementLookup, ctestX, ctestY);
          if (f > bestFitness) {
            bestFitness = f;
            bestStepSize = interval;
          }
        }
      }
      if (bestStepSize == 0)
        break;

      cx = (int)(cx + bestStepSize * gx);
      cy = (int)(cy + bestStepSize * gy);

      if(!silentMode) line(debugImage, Point(coldX, coldY), Point(cx, cy), Scalar(255, 0, 0), 1, 8, 0);
      
      if(cx == coldX && cy == coldY)
        break;
    }
    centerPoints[i] = Point(cx, cy);
    centerFitness[i] = fitness(image.cols, image.rows, grad_x, grad_y, displacementLookup, cx, cy);
  }
  
  // Find the center with maximum fitness
  double maximumFitness = centerFitness[0];
  Point maximumCenter = centerPoints[0];
  for(int i = 1; i < m; i++) {
    if(centerFitness[i] > maximumFitness) {
      maximumFitness = centerFitness[i];
      maximumCenter = centerPoints[i];
    }
  }

  // DEBUG drawing
  if(!silentMode) {
    for(int i = 0; i < m; i++) {
      Point c(highest_magnitude_pixels_x[i], highest_magnitude_pixels_y[i]);
      line(debugImage, Point(c.x - 5, c.y), Point(c.x + 5, c.y), Scalar(0, 255, 0), 1, 8, 0);
      line(debugImage, Point(c.x, c.y - 5), Point(c.x, c.y + 5), Scalar(0, 255, 0), 1, 8, 0);
    }
    imshow("debug", debugImage);
    
    showNormalizedImage(grad_x, "grad_x");
    showNormalizedImage(grad_y, "grad_y");
    showNormalizedImage(fitnessLookup, "fitness");
  }

  centers = new Point[1];
  centers[0] = maximumCenter;
  return 1;
}
