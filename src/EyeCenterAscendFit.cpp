#include "Utils.h"
#include "Gradient.h"
#include "EyeCenterAscendFit.h"
#include <stdio.h>
#include <math.h>  

using namespace cv;

float getFitnessLookup(Mat& image, Mat& grad_x, Mat& grad_y, Mat& fitnessLookup, Mat& displacementLookup, int x, int y) {
  x = min(image.cols - 1, max(0, x));
  y = min(image.rows - 1, max(0, y));
  float val = fitnessLookup.at<float>(y, x);
  if(val < 0) {
    val = fitness(image, grad_x, grad_y, displacementLookup, x, y);
    fitnessLookup.at<float>(y, x) = val;
  }
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
  const int tmax = 300;
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

  double maxMagnitude = -1, minMagnitude = -1;
  
  // Find the m highest magnitudes
  double highest_magnitudes[m];
  for(int i = 0; i < m; i++) {
    highest_magnitudes[i] = -1;
  }
  Point highest_magnitude_pixels[m];
  for(int y = 0; y < grad_x.rows; y++) {
    for(int x = 0; x < grad_x.cols; x++) {
      double gx = grad_x.at<float>(y, x);
      double gy = grad_y.at<float>(y, x);
      double mag = sqrt(gx * gx + gy * gy);
      if(mag > maxMagnitude)
        maxMagnitude = mag;
      if(mag < minMagnitude || minMagnitude == -1)
        minMagnitude = mag;

      int index = -1;
      while(mag > highest_magnitudes[index + 1] && index < m - 1) {
        index += 1;
      }
      if(index > -1) {
        for(int i = 0; i < index; i++) {
          highest_magnitudes[i] = highest_magnitudes[i + 1];
          highest_magnitude_pixels[i] = Point(highest_magnitude_pixels[i + 1]);
        }
        highest_magnitudes[index] = mag;
        highest_magnitude_pixels[index] = Point(x, y);
      }   
    }
  }

  float gx, gy, length;
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

  Mat displacementLookup = buildDisplacementLookup(image.cols, image.rows);
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
  Point2f d_i, g_i;
  for(int i = 0; i < m; i++) {
    Point2f c = highest_magnitude_pixels[i];
    Point2f c_old;
    for(int j = 0; j < tmax; j++) {
      c_old = Point2f(c.x, c.y);
      
      float left = getFitnessLookup(image, grad_x, grad_y, fitnessLookup, displacementLookup, (int)c.x - 1, (int)c.y);
      float right = getFitnessLookup(image, grad_x, grad_y, fitnessLookup, displacementLookup, (int)c.x + 1, (int)c.y);
      float top = getFitnessLookup(image, grad_x, grad_y, fitnessLookup, displacementLookup, (int)c.x, (int)c.y - 1);
      float bottom = getFitnessLookup(image, grad_x, grad_y, fitnessLookup, displacementLookup, (int)c.x, (int)c.y + 1);
      Point2f g((right - left) * 0.5f, (bottom - top) * 0.5f);
      length = sqrt(g.x * g.x + g.y * g.y);
      if (length > 0) {
        g /= length;
      }
      //std::cout << left << "," << right << "," << top << "," << bottom << " - " << g << std::endl;

      // evaluate J(c)
      double bestFitness = -1, bestStepSize = 0;
      for (int h = 0; h < stepSizeCount; h++) {
        interval = stepSizes[h];
        
        Point cTemp((int) (c.x + interval * g.x), (int) (c.y + interval * g.y));
        if (!bordersReached(cTemp.x, cTemp.y, image.cols, image.rows)) {
          double f = fitness(image, grad_x, grad_y, displacementLookup, cTemp.x, cTemp.y);
          //std::cout << "\t\tCheck Stepsize: " << interval << ", fitness: " << f << std::endl;
          if (f > bestFitness) {
            //std::cout << "\t\t\tConsidered" << std::endl;
            bestFitness = f;
            bestStepSize = interval;
          }
        }
      }
      if (bestStepSize <= 0) {
        break;
      }
      //std::cout << "\tStepsize: " << bestStepSize << std::endl;
      c.x += bestStepSize * g.x;
      c.y += bestStepSize * g.y;

      if(!silentMode) line(debugImage, c_old, c, Scalar(255, 0, 0), 1, 8, 0);
      
      Point2f diff = c - c_old;
      double length = sqrt(diff.x * diff.x + diff.y * diff.y);
      
      //std::cout << "\tNew Center: " << c << ", distance: " << length << std::endl;
      
      if(bordersReached(c.x, c.y, image.cols, image.rows) || length <= sigma)
        break;
    }
    centerPoints[i] = c;
    centerFitness[i] = fitness(image, grad_x, grad_y, displacementLookup, c.x, c.y);
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
      Point c = highest_magnitude_pixels[i];
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
