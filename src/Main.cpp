#include "EyeCenterNaive.h"
#include "EyeCenterAscend.h"
#include "EyeCenterAscendFit.h"
#include "EyeCenterEvolAlg.h"
#include <algorithm>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

const String MODE_NAIVE = "naive";
const String MODE_ASCEND = "ascend";
const String MODE_ASCEND_FIT = "ascendfit";
const String MODE_EVOL = "evol";

char* getArg(char ** begin, char ** end, const std::string & option) {
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) {
    return *itr;
  }
  return 0;
}

bool argExists(char** begin, char** end, const std::string& option) {
  return std::find(begin, end, option) != end;
}

int main(int argc, char** argv) {
  if (argc < 2 || argExists(argv, argv + argc, "-h") || argExists(argv, argv + argc, "--help")) {
    std::cout << "Usage: EyeCenter [args]" << std::endl;
    std::cout << "\t-h, --help\tPrint this help message." << std::endl;
    std::cout << "\t-s\t\tActivate silent mode (good for evaluation)." << std::endl;
    std::cout << "\t-r\t\tResize the image to 100 width (for fast testing)." << std::endl;
    std::cout << "\t-m [MODE]\tAnalysis mode [naive, ascend, ascendfit, evol]." << std::endl;
    std::cout << "\t-i [FILE]\tImage file to analyze." << std::endl;
    return 0;
  }

  String inputFilepath;
  if (argExists(argv, argv + argc, "-i")) {
    inputFilepath = getArg(argv, argv + argc, "-i");
  }

  String mode = MODE_NAIVE;
  if (argExists(argv, argv + argc, "-m")) {
    mode = getArg(argv, argv + argc, "-m");
  }
  if (mode != MODE_NAIVE && mode != MODE_ASCEND && mode != MODE_ASCEND_FIT && mode != MODE_EVOL) {
    std::cout << "Unknown mode " << mode << std::endl;
    return -1;
  }

  bool silentMode = argExists(argv, argv + argc, "-s");

  Mat image;
  image = imread(inputFilepath, 1);
  if (!image.data) {
    std::cout << "Failed to load image " << inputFilepath << std::endl;
    return -1;
  }

  // For faster testing reduce the image size
  if (argExists(argv, argv + argc, "-r")) {
    int height = 100;
    Size scale((int)(image.cols * (height / (float)image.rows)), height);
    resize(image, image, scale);
  }

  // Find all eye centers
  Point* eyeCenters;
  int eyeCenterCount;
  int64 ticks = getTickCount();
  if (mode == MODE_NAIVE) {
    eyeCenterCount = EyeCenterNaive::findEyeCenters(image, eyeCenters, silentMode);
  } else if (mode == MODE_ASCEND) {
    eyeCenterCount = EyeCenterAscend::findEyeCenters(image, eyeCenters, silentMode);
  } else if (mode == MODE_ASCEND_FIT) {
    eyeCenterCount = EyeCenterAscendFit::findEyeCenters(image, eyeCenters, silentMode);
  } else if (mode == MODE_EVOL) {
    eyeCenterCount = EyeCenterEvolAlg::findEyeCenters(image, eyeCenters, silentMode);
  }
  // Print out the time used for the detection mode
  std::cout << ((getTickCount() - ticks) / getTickFrequency()) << std::endl;
  
  // Print the results
  for (int i = 0; i < eyeCenterCount; i++) {
    std::cout << eyeCenters[i].x << "\t" << eyeCenters[i].y << std::endl;
  }

  // Show result
  if (!silentMode) {
    for (int i = 0; i < eyeCenterCount; i++) {
      line(image, Point2i(0, eyeCenters[i].y), Point2i(image.cols - 1, eyeCenters[i].y),
           Scalar(0, 0, 255), 1, 8, 0);
      line(image, Point2i(eyeCenters[i].x, 0), Point2i(eyeCenters[i].x, image.rows - 1),
           Scalar(0, 0, 255), 1, 8, 0);
    }
    imshow("image", image);
  }

  waitKey(0);
  return 0;
}
