//#include "Utils.h"
#include "Gradient.h"
#include "EyeCenterAscend.h"
#include <stdio.h>

using namespace cv;

bool bordersReached(Point c, int w, int h) {
  return c.x <= 0 || c.x >= w - 1 || c.y <= 0 || c.y >= h - 1;
}

double fitness(Mat& image, Mat& grad_x, Mat& grad_y, Point c) {
  double fitness = 0, length, dot;
  Point2f d, g;
  for(int y = 0; y < image.rows; y++) {
    for(int x = 0; x < image.cols; x++) {
      // Normalized distance vector
      d = Point2f(x - c.x, y - c.y);
      length = sqrt(d.x * d.x + d.y * d.y);
      if(length > 0) {
        d /= length;
      }
      // Normalized gradient vector
      g = Point2f(grad_x.at<float>(y, x), grad_y.at<float>(y, x));
      dot = d.dot(g);
      fitness += dot * dot;
    }
  }
  int N = image.cols * image.rows;
  return fitness / N;
}

int EyeCenterAscend::findEyeCenters(Mat& image, Point*& centers) {
  //GaussianBlur(image, image, Size(5, 5), 0, 0, BORDER_DEFAULT);

  Mat grey(image.size(), CV_8UC1);
  cvtColor(image, grey, CV_RGB2GRAY);

  Mat grad_x, grad_y;
  calculateGradients(Four_Neighbor, grey, grad_x, grad_y);

  const int m = 100;
  const int tmax = 20;
  const int N = image.cols * image.rows;
  const double sigma = 0.001;

  // Find the m highest magnitudes
  double highest_magnitudes[m];
  Point highest_magnitude_pixels[m];
  for(int y = 0; y < grad_x.rows; y++) {
    for(int x = 0; x < grad_x.cols; x++) {
      double gx = grad_x.at<float>(y, x);
      double gy = grad_y.at<float>(y, x);
      double mag = sqrt(gx * gx + gy * gy);
      
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

  Point centerPoints[m];
  double centerFitness[m];
  for(int i = 0; i < m; i++) {
    Point c = highest_magnitude_pixels[i];
    Point c_old;
    for(int j = 0; j < tmax; j++) {
      c_old = Point(c.x, c.y);
      Point g(0, 0);
      for(int y = 0; y < image.rows; y++) {
        for(int x = 0; x < image.cols; x++) {
          // TODO
        }
      }
      g = g * 2 / N;
      
      double s = 0; // TODO
      
      c += s * g;
      
      Point diff = c - c_old;
      double length = sqrt(diff.x * diff.x + diff.y * diff.y);
      
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

  centers = new Point[1];
  centers[0] = maximumCenter;
  return 1;
}
