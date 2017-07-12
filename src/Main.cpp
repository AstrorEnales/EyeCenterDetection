#include "EyeCenterNaive.h"
#include "EyeCenterAscend.h"
#include "EyeCenterAscendPaul.h"
#include "EyeCenterAscendFit.h"
#include "EyeCenterEvolAlg.h"
#include <algorithm>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>

using namespace cv;

const String MODE_NAIVE = "naive";
const String MODE_ASCEND = "ascend";
const String MODE_ASCEND_FIT = "ascendfit";
const String MODE_EVOL = "evol";
const String MODE_PAUL = "paul";

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

int runAnalysis(Mat& image, String mode, bool silentMode, bool logResultsAndTime, Point*& eyeCenters) {
  // Find all eye centers
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
  } else if (mode == MODE_PAUL) {
    eyeCenterCount = EyeCenterAscendPaul::findEyeCenters(image, eyeCenters, silentMode);
  }

  if (logResultsAndTime) {
    // Print out the time used for the detection mode
    std::cout << ((getTickCount() - ticks) / getTickFrequency()) << std::endl;
  
    // Print the results
    for (int i = 0; i < eyeCenterCount; i++) {
      std::cout << eyeCenters[i].x << "\t" << eyeCenters[i].y << std::endl;
    }
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
  return eyeCenterCount;
}

int main(int argc, char** argv) {
  if (argc < 2 || argExists(argv, argv + argc, "-h") || argExists(argv, argv + argc, "--help")) {
    std::cout << "Usage: EyeCenter [args]" << std::endl;
    std::cout << "\t-h, --help\tPrint this help message." << std::endl;
    std::cout << "\t-s\t\tActivate silent mode (good for evaluation)." << std::endl;
    std::cout << "\t-r\t\tResize the image to 100 width (for fast testing)." << std::endl;
    std::cout << "\t-m [MODE]\tAnalysis mode [naive, ascend, ascendfit, evol, paul]." << std::endl;
    std::cout << "\t-i [FILE]\tImage file to analyze." << std::endl;
    std::cout << "\t-c [SOURCE]\tLive camera feed." << std::endl;
    std::cout << "\t-f\t\tFace mode (detect faces before analysis)." << std::endl;
    std::cout << "\t-fc [PATH]\tClassifier path." << std::endl;
    return 0;
  }

  String inputFilepath;
  if (argExists(argv, argv + argc, "-i")) {
    inputFilepath = getArg(argv, argv + argc, "-i");
  }

  String classifierPath;
  if (argExists(argv, argv + argc, "-fc")) {
    classifierPath = getArg(argv, argv + argc, "-fc");
  } else {
    classifierPath = ".";
  }
  
  int cameraSource;
  bool liveCameraMode = argExists(argv, argv + argc, "-c");
  if (liveCameraMode) {
    String sourceString = getArg(argv, argv + argc, "-c");
    try {
      cameraSource = std::stoi(sourceString);
    }
    catch(Exception ex) {
      std::cout << "Failed to parse camera source. Using default 0." << std::endl;
      cameraSource = 0;
    }
  }

  String mode = MODE_NAIVE;
  if (argExists(argv, argv + argc, "-m")) {
    mode = getArg(argv, argv + argc, "-m");
  }
  if (mode != MODE_NAIVE && mode != MODE_ASCEND && mode != MODE_ASCEND_FIT && mode != MODE_EVOL && mode != MODE_PAUL) {
    std::cout << "Unknown mode " << mode << std::endl;
    return -1;
  }

  bool silentMode = argExists(argv, argv + argc, "-s");
  bool faceMode = argExists(argv, argv + argc, "-f");

  if (liveCameraMode) {
    VideoCapture cap(cameraSource);
    if (!cap.isOpened()) {
      std::cout << "Unable to connect to camera" << std::endl;
      return -1;
    }
    
    CascadeClassifier faceCascade;
    faceCascade.load(classifierPath + "/haarcascade_frontalface_alt2.xml");
    CascadeClassifier eyeCascade;
    eyeCascade.load(classifierPath + "/haarcascade_eye.xml");
    std::vector<Rect> faces;
    std::vector<Rect> eyes;
    Mat image;
    while(true) {
      cap >> image;

      faceCascade.detectMultiScale(image, faces, 1.1, 2, CV_HAAR_SCALE_IMAGE, Size(30, 30));
      for(unsigned int i = 0; i < faces.size(); i++) {
        Mat face = image(faces.at(i));
        eyeCascade.detectMultiScale(face, eyes, 1.1, 3, CV_HAAR_SCALE_IMAGE, Size(30, 30));
        for(unsigned int j = 0; j < eyes.size(); j++) {
          //Point center(faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5);
          //ellipse(image, center, Size(eyes[j].width*0.5, eyes[j].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
        
          Mat eye = face(eyes.at(j));
          Point* eyeCenters;
          runAnalysis(eye, mode, true, false, eyeCenters);
          eyeCenters[0].x += faces[i].x + eyes[j].x;
          eyeCenters[0].y += faces[i].y + eyes[j].y;
          line(image, eyeCenters[0] - Point(5, 0), eyeCenters[0] + Point(5, 0), Scalar(0, 0, 255), 1, 8, 0);
          line(image, eyeCenters[0] - Point(0, 5), eyeCenters[0] + Point(0, 5), Scalar(0, 0, 255), 1, 8, 0);
        }
      }
      imshow("image", image);
      if(cv::waitKey(30)  == ' ') {
        break;
      }
    }
    destroyWindow("image");

  } else {
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
    
    Point* eyeCenters;
    runAnalysis(image, mode, silentMode, true, eyeCenters);
  }

  waitKey(0);
  return 0;
}
