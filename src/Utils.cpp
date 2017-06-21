#include "Utils.h"

void showNormalizedImage(const Mat& img, std::string name) {
  Mat imgNormalized;
  normalize(img, imgNormalized, 1, 0, NORM_MINMAX);
  imshow(name, imgNormalized);
}

bool bordersReached(Point c, int w, int h) {
  return c.x <= 0 || c.x >= w - 1 || c.y <= 0 || c.y >= h - 1;
}

Mat buildDisplacementLookup(int w, int h) {
  float length;
  Mat displacementLookup(Size(w * 2, h * 2), CV_32FC2);
  Point displacementLookupCenter(w, h);
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
  return displacementLookup;
}

double fitness(Mat& image, Mat& grad_x, Mat& grad_y, Mat& displacementLookup, int cx, int cy) {
  double fitness = 0;
  float length, dot;
  Point2f d;
  for(int y = 0; y < image.rows; y++) {
    float* grad_x_ptr = grad_x.ptr<float>(y);
    float* grad_y_ptr = grad_y.ptr<float>(y);
    Point2f* displacement_lookup_ptr = displacementLookup.ptr<Point2f>(image.rows + y - cy);
    for(int x = 0; x < image.cols; x++) {
      d = displacement_lookup_ptr[image.cols + x - cx];
      dot = max(0.0f, d.x * grad_x_ptr[x] + d.y * grad_y_ptr[x]);
      fitness += dot * dot;
    }
  }
  int N = image.cols * image.rows;
  return fitness / N;
}
