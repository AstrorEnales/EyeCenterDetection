#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
/*
int main(int argc, char** argv) {
    if (argc != 2) {
        printf("usage: EyeCenter <Image_Path>\n");
        return -1;
    }

    Mat image;
    image = imread(argv[1], 1);
    if (!image.data) {
        printf("No image data\n");
        return -1;
    }

    GaussianBlur(image, image, Size(5, 5), 0, 0, BORDER_DEFAULT);

    Mat grey(image.size(), CV_8UC1);
    cvtColor(image, grey, CV_RGB2GRAY);

    Mat grad_x, grad_y;
    Scharr(grey, grad_x, CV_32FC1, 1, 0);
    Scharr(grey, grad_y, CV_32FC1, 0, 1);

    Mat magnitude, direction;
    bool useDegree = true;
    cartToPolar(grad_x, grad_y, magnitude, direction, useDegree);

    Mat scaled;
    convertScaleAbs(direction, scaled);
    imshow("direction", scaled);

    int m = 100;
    int tmax = 20;
    int N = image.cols * image.rows;

    float highest_magnitudes[m];
    Point2i highest_magnitude_pixels[m];
    for(int y = 0; y < magnitude.rows; y++) {
      for(int x = 0; x < magnitude.cols; x++) {
        float mag = abs(magnitude.at<float>(x, y));
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
          highest_magnitude_pixels[index] = Point2i(x, y);
        }
      }
    }

    for(int i = 0; i < m; i++) {
      Point2i c = highest_magnitude_pixels[i];
      Point2i c_old;
      for(int j = 0; j < tmax; j++) {
        c_old = c;
        float g = 0;
        for(int y = 0; y < magnitude.rows; y++) {
          for(int x = 0; x < magnitude.cols; x++) {
            
          }
        }
        g = g * 2 / N;
      }
    }

    for(int i = 0; i < m; i++) {
      line(image, highest_magnitude_pixels[i] - Point2i(5, 0), highest_magnitude_pixels[i] + Point2i(5, 0), Scalar(0, 0, 255), 1, 8, 0);
      line(image, highest_magnitude_pixels[i] - Point2i(0, 5), highest_magnitude_pixels[i] + Point2i(0, 5), Scalar(0, 0, 255), 1, 8, 0);
    }
    imshow("highest magnitudes", image);

    waitKey(0);

    return 0;
}
*/
