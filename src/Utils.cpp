#include "Utils.h"

void showNormalizedImage(const Mat& img, std::string name) {
  Mat imgNormalized;
  normalize(img, imgNormalized, 1, 0, NORM_MINMAX);
  imshow(name, imgNormalized);
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
  int i, x;
  cy = image.rows - cy;
  cx = image.cols - cx;
  for(int y = 0; y < image.rows; y++) {
    const float* grad_x_ptr = grad_x.ptr<float>(y);
    const float* grad_y_ptr = grad_y.ptr<float>(y);
    const float* displacement_lookup_ptr = displacementLookup.ptr<float>(cy + y);
    i = cx * 2;
    for(x = 0; x < image.cols; x++) {
      i += 2;
      const float& dx = displacement_lookup_ptr[i];
      const float& dy = displacement_lookup_ptr[i+1];
      dot = dx * grad_x_ptr[x] + dy * grad_y_ptr[x];
      if(dot > 0)
        fitness += dot;
    }
  }
  const int N = image.cols * image.rows;
  return fitness / N;
}
