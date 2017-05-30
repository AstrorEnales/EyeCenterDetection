#include "EyeCenterNaive.h"
#include "EyeCenterAscend.h"
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cout << "usage: EyeCenter <Image_Path>" << std::endl;
    return -1;
  }

  Mat image;
  image = imread(argv[1], 1);
  if (!image.data) {
    std::cout << "No image data" << std::endl;
    return -1;
  }

  // For faster testing reduce the image size
  int height = 100;
  Size scale((int)(image.cols * (height / (float)image.rows)), height);
  resize(image, image, scale);

  // Find all eye centers
  Point* eyeCenters;
  int eyeCenterCount;
  eyeCenterCount = EyeCenterNaive::findEyeCenters(image, eyeCenters);
  //eyeCenterCount = EyeCenterAscend::findEyeCenters(image, eyeCenters);

  // Print the results and draw crosses into the image
  for (int i = 0; i < eyeCenterCount; i++) {
    std::cout << "target: " << eyeCenters[i] << std::endl;
    line(image, Point2i(0, eyeCenters[i].y), Point2i(image.cols - 1, eyeCenters[i].y),
         Scalar(0, 0, 255), 1, 8, 0);
    line(image, Point2i(eyeCenters[i].x, 0), Point2i(eyeCenters[i].x, image.rows - 1),
         Scalar(0, 0, 255), 1, 8, 0);
  }

  // Show result
  imshow("image", image);

  waitKey(0);
  return 0;
}
