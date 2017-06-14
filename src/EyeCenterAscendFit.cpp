#include "Utils.h"
#include "Gradient.h"
#include "EyeCenterAscendFit.h"
#include <stdio.h>
#include <math.h>  

using namespace cv;

int EyeCenterAscendFit::findEyeCenters(Mat& image, Point*& centers, bool silentMode) {
  GaussianBlur(image, image, Size(5, 5), 0, 0, BORDER_DEFAULT);

  Mat grey(image.size(), CV_8UC1);
  cvtColor(image, grey, CV_RGB2GRAY);

  Mat debugImage;
  if(!silentMode) debugImage = image.clone();

  Mat fitnessLookup(image.size(), CV_32FC1, float(-1));

  Mat grad_x, grad_y;
  calculateGradients(Four_Neighbor, grey, grad_x, grad_y);

  const int m = 1;
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

  Point centerPoints[m];
  double centerFitness[m];
  Point2f d_i, g_i;
  float n, e_i, length;
  for(int i = 0; i < m; i++) {
    // getInitialCenter(i)
    Point2f c = highest_magnitude_pixels[i];
    //std::cout << "Start center: " << c << std::endl;
    Point2f c_old;
    for(int j = 0; j < tmax; j++) {
      //std::cout << "\tIteration: " << j << std::endl;
      c_old = Point2f(c.x, c.y);
      
      for(int y = -10; y <= 10; y++)
        for(int x = -10; x <= 10; x++)
          if(fitnessLookup.at<float>((int)c.y + y, (int)c.x + x) < 0)
            fitnessLookup.at<float>((int)c.y + y, (int)c.x + x) = fitness(image, grad_x, grad_y, Point((int)c.y + y, (int)c.x + x));
      
      float left = fitnessLookup.at<float>((int)c.y, (int)c.x - 1);
      float right = fitnessLookup.at<float>((int)c.y, (int)c.x + 1);
      float top = fitnessLookup.at<float>((int)c.y - 1, (int)c.x);
      float bottom = fitnessLookup.at<float>((int)c.y + 1, (int)c.x);
      if(left < 0) {
        fitnessLookup.at<float>((int)c.y, (int)c.x - 1) = left = fitness(image, grad_x, grad_y, Point((int)c.y, (int)c.x - 1));
      }
      if(right < 0) {
        fitnessLookup.at<float>((int)c.y, (int)c.x + 1) = right = fitness(image, grad_x, grad_y, Point((int)c.y, (int)c.x + 1));
      }
      if(top < 0) {
        fitnessLookup.at<float>((int)c.y - 1, (int)c.x) = top = fitness(image, grad_x, grad_y, Point((int)c.y - 1, (int)c.x));
      }
      if(bottom < 0) {
        fitnessLookup.at<float>((int)c.y + 1, (int)c.x) = bottom = fitness(image, grad_x, grad_y, Point((int)c.y + 1, (int)c.x));
      }
      Point2f g((right - left) * 0.5f, (bottom - top) * 0.5f);
      length = sqrt(g.x * g.x + g.y * g.y);
      if (length > 0) {
        g /= length;
      }
      //std::cout << "\tG: " << g << std::endl;

      // evaluate J(c)
      double bestFitness = -1, bestStepSize = 0;
      for (int h = 0; h < stepSizeCount; h++) {
        interval = stepSizes[h];
        
        Point cTemp((int) (c.x + interval * g.x), (int) (c.y + interval * g.y));
        if (!bordersReached(cTemp, image.cols, image.rows)) {
          double f = fitness(image, grad_x, grad_y, cTemp);
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
      
      if(bordersReached(c, image.cols, image.rows) || length <= sigma)
        break;
    }
    centerPoints[i] = c;
    centerFitness[i] = fitness(image, grad_x, grad_y, c);
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

    showNormalizedImage(fitnessLookup, "fitness");
  }

  centers = new Point[1];
  centers[0] = maximumCenter;
  return 1;
}
