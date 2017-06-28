#include "Utils.h"
#include "Gradient.h"
#include "EyeCenterAscend.h"
#include <stdio.h>
#include <math.h>  

using namespace cv;

int EyeCenterAscend::findEyeCenters(Mat& image, Point*& centers, bool silentMode) {
  //GaussianBlur(image, image, Size(5, 5), 0, 0, BORDER_DEFAULT);

  Mat grey(image.size(), CV_8UC1);
  cvtColor(image, grey, CV_RGB2GRAY);

  Mat debugImage;
  if(!silentMode) debugImage = image.clone();

  Mat grad_x, grad_y;
  calculateGradients(Four_Neighbor, grey, grad_x, grad_y);

  const int m = 50;
  const int tmax = 30;
  const int N = image.cols * image.rows;
  const double sigma = 0.001;
  const int stepSizeCount = 10;
  const double stepIntervalMin = pow(10, -2);
  const double stepIntervalMax = pow(10, 3);

  double interval;
  double stepSizes[stepSizeCount];
  // for n = 10, exponentially increase stepsize [10⁻², 10⁵]
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

  Mat displacementLookup;
  buildDisplacementLookup(displacementLookup, image.cols, image.rows);

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
      Point2d g(0, 0);
      for(int y = 0; y < image.rows; y++) {
        float* grad_ptr_x = grad_x.ptr<float>(y);
        float* grad_ptr_y = grad_y.ptr<float>(y);
        for(int x = 0; x < image.cols; x++) {
          if (x == (int)c.x && y == (int)c.y) {
            continue;
          }
          // computeGradient(c, X, G)
          g_i = Point2f(grad_ptr_x[x], grad_ptr_y[x]);
          length = sqrt(g_i.x * g_i.x + g_i.y * g_i.y);
          if(length > 0)
            g_i /= length;
          d_i = Point2f(x - c.x, y - c.y);
          n = d_i.x * d_i.x + d_i.y * d_i.y; // n * n so sqrt is unnecessary
          e_i = d_i.dot(g_i);
          //g.x += (d_i.x * (e_i * e_i) - g.x * e_i * n) / (n * n);
          //g.y += (d_i.y * (e_i * e_i) - g.y * e_i * n) / (n * n);
          g.x += (d_i.x * (e_i * e_i)) / (n * n);
          g.y += (d_i.y * (e_i * e_i)) / (n * n);
        }
      }
      g = g * 2 / N;
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
        if (!bordersReached(cTemp.x, cTemp.y, image.cols, image.rows)) {
          double f = fitness(image.cols, image.rows, grad_x, grad_y, displacementLookup, cTemp.x, cTemp.y);
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
      
      if(length <= sigma)
        break;
    }
    centerPoints[i] = c;
    centerFitness[i] = fitness(image.cols, image.rows, grad_x, grad_y, displacementLookup, c.x, c.y);
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
  }

  centers = new Point[1];
  centers[0] = maximumCenter;
  return 1;
}
