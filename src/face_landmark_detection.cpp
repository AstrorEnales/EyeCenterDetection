#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <iostream>
#include "face_landmark_detection.h"

using namespace dlib;
using namespace std;
using namespace cv;

void FD::getEyes(Mat& image, String classifierPath, bool setNull, Mat& leftEye, Mat& rightEye,
                 Point& leftEyeOffset, Point& rightEyeOffset) {
	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor sp;
	deserialize(classifierPath + "/shape_predictor_68_face_landmarks.dat") >> sp;

  array2d<rgb_pixel> dlibImage;
  dlib::assign_image(dlibImage, dlib::cv_image<bgr_pixel>(image));

	// Make the image larger so we can detect small faces.
	//pyramid_up(dlibImage); //TODO: evaluate if necessary

	std::vector<dlib::rectangle> dets = detector(dlibImage);
	// detect the rectangle for the eyes
	int min_x_l = image.cols;
	int max_x_l = 0;
	int min_y_l = image.rows;
	int max_y_l = 0;
	int min_x_r = image.cols;
	int max_x_r = 0;
	int min_y_r = image.rows;
	int max_y_r = 0;
	for (unsigned long j = 0; j < dets.size(); ++j) {
		full_object_detection shape = sp(dlibImage, dets[j]);
		// left eye
		for (int k = 36; k <= 41; k++) {
			int x = (int)shape.part(k).x();
			int y = (int)shape.part(k).y();
			min_x_l = min(min_x_l, x);
			max_x_l = max(max_x_l, x);
			min_y_l = min(min_y_l, y);
			max_y_l = max(max_y_l, y);
		}
		// right eye
		for (int k = 42; k <= 47; k++) {
			int x = (int)shape.part(k).x();
			int y = (int)shape.part(k).y();
			min_x_r = min(min_x_r, x);
			max_x_r = max(max_x_r, x);
			min_y_r = min(min_y_r, y);
			max_y_r = max(max_y_r, y);
		}

    Mat destination;
		if (setNull) {
			// Create a copy and draw a black rectangle at the eyes. Using the landmarks the
			// eye polygons are filled with white. After copying the original image into
			// the mask, only the eye polygon remains and the rest is black.
			Mat copy = image.clone();

			// Make rectangle black for eyes
			cv::rectangle(copy, Rect(min_x_l, min_y_l, max_x_l - min_x_l, max_y_l - min_y_l), Scalar(0, 0, 0), CV_FILLED);
			cv::rectangle(copy, Rect(min_x_r, min_y_r, max_x_r - min_x_r, max_y_r - min_y_r), Scalar(0, 0, 0), CV_FILLED);

			// Make polygons for eyes
			std::vector<Point> fillSinglePolyL;
			std::vector<Point> fillSinglePolyR;
			for (int k = 36; k <= 41; k++) {
				fillSinglePolyL.push_back(Point((int)shape.part(k).x(), (int)shape.part(k).y()));
			}
			for (int k = 42; k <= 47; k++) {
				fillSinglePolyR.push_back(Point((int)shape.part(k).x(), (int)shape.part(k).y()));
			}
			std::vector<std::vector<Point>> fillContPolyL;
			fillContPolyL.push_back(fillSinglePolyL);
			std::vector<std::vector<Point>> fillContPolyR;
			fillContPolyR.push_back(fillSinglePolyR);
			cv::fillPoly(copy, fillContPolyL, Scalar(255, 255, 255));
			cv::fillPoly(copy, fillContPolyR, Scalar(255, 255, 255));

			// Merge original with polygon
      image.copyTo(destination, copy);
    }
    else {
      destination = image.clone();
    }
		// Cut the eyes
		rightEye = Mat(destination, Rect(min_x_l, min_y_l, max_x_l - min_x_l, max_y_l - min_y_l));
		leftEye = Mat(destination, Rect(min_x_r, min_y_r, max_x_r - min_x_r, max_y_r - min_y_r));
    leftEyeOffset = Point(min_x_l, min_y_l);
    rightEyeOffset = Point(min_x_r, min_y_r);
    // TODO: currently just the first face
    break;
	}
}