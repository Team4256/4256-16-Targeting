#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
using namespace cv;
using namespace std;

#pragma once
void inRangeRGB(InputArray src, OutputArray dst, Scalar lowerHSVBounds, Scalar upperHSVBounds, bool makeOutputBinaryMask);
void inRangeHLS(InputArray src, OutputArray dst, Scalar lowerHSVBounds, Scalar upperHSVBounds, bool makeOutputBinaryMask);
void inRangeHSV(InputArray src, OutputArray dst, Scalar lowerHSVBounds, Scalar upperHSVBounds, bool makeOutputBinaryMask);
void dilate_erode(InputArray src, OutputArray dst, int itterations = 1);
void dilate(InputArray src, OutputArray dst, int itterations = 1);
void erode(InputArray src, OutputArray dst, int itterations = 1);
void drawCircle(Mat dst, vector<Vec3f> circles);

void blobDetector(Mat src);