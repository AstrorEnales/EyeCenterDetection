#include "Utils.h"

void showNormalizedImage(const Mat& img, std::string name) {
  Mat imgNormalized;
  normalize(img, imgNormalized, 1, 0, NORM_MINMAX);
  imshow(name, imgNormalized);
}

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
