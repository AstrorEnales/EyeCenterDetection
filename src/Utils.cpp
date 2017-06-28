#include "Utils.h"

void showNormalizedImage(const Mat& img, std::string name) {
  Mat imgNormalized;
  normalize(img, imgNormalized, 1, 0, NORM_MINMAX);
  imshow(name, imgNormalized);
}

void buildDisplacementLookup(Mat& displacementLookup, int w, int h) {
  float length, dx, dy;
  int x, i;
  displacementLookup = Mat(Size(w * 2, h * 2), CV_32FC2, float(0));
  Point displacementLookupCenter(w, h);
  for(int y = 0; y < displacementLookup.rows; y++) {
    float* displacement_lookup_ptr = displacementLookup.ptr<float>(y);
    dy = y - displacementLookupCenter.y;
    for(x = 0, i = 0; x < displacementLookup.cols; x++) {
      // Normalized distance vector
      dx = x - displacementLookupCenter.x;
      length = sqrt(dx * dx + dy * dy);
      if(length > 0) {
        displacement_lookup_ptr[i++] = dx / length;
        displacement_lookup_ptr[i++] = dy / length;
      }
    }
  }
}

double fitness(const int cols, const int rows, Mat& grad_x, Mat& grad_y, Mat& displacementLookup, int cx, int cy) {
  double fitness = 0;
  float length, dot;
  int i, x;
  cy = rows - cy;
  cx = cols - cx;
  for(int y = 0; y < rows; y++) {
    const float* grad_x_ptr = grad_x.ptr<float>(y);
    const float* grad_y_ptr = grad_y.ptr<float>(y);
    const float* displacement_lookup_ptr = displacementLookup.ptr<float>(cy + y);
    i = cx * 2;
    for(x = 0; x < cols; x++) {
      const float& dx = displacement_lookup_ptr[i++];
      const float& dy = displacement_lookup_ptr[i++];
      dot = dx * grad_x_ptr[x] + dy * grad_y_ptr[x];
      if(dot > 0)
        fitness += dot;
    }
  }
  const int N = cols * rows;
  return fitness / N;
}
