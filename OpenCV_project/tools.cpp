#include "opencv2/opencv.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

void inRangeRGB(InputArray src, OutputArray dst, Scalar lowerHSVBounds, Scalar upperHSVBounds, bool makeOutputBinaryMask) {
	inRange(src, lowerHSVBounds, upperHSVBounds, dst);

	if (!makeOutputBinaryMask) {
		Mat output;
		src.copyTo(output, dst);
		output.copyTo(dst);
	}
}

void inRangeHLS(InputArray src, OutputArray dst, Scalar lowerHSVBounds, Scalar upperHSVBounds, bool makeOutputBinaryMask) {
	cvtColor(src, dst, COLOR_RGB2HLS);
	inRange(dst, lowerHSVBounds, upperHSVBounds, dst);

	if (!makeOutputBinaryMask) {
		Mat output;
		src.copyTo(output, dst);
		output.copyTo(dst);
	}
}

void inRangeHSV(InputArray src, OutputArray dst, Scalar lowerHSVBounds, Scalar upperHSVBounds, bool makeOutputBinaryMask) {
	cvtColor(src, dst, COLOR_BGR2HSV);
	inRange(dst, lowerHSVBounds, upperHSVBounds, dst);

	if (!makeOutputBinaryMask) {
		Mat output;
		src.copyTo(output, dst);
		output.copyTo(dst);
	}
}

void dilate_erode(InputArray src, OutputArray dst, int itterations = 1) {
	int erosion_size = 1;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
		cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		cv::Point(erosion_size, erosion_size));
	erode(src, dst, element, Point(-1, -1), itterations);
	dilate(src, dst, element, Point(-1, -1), itterations);
	dilate(src, dst, element, Point(-1, -1), itterations);
	erode(src, dst, element, Point(-1, -1), itterations);
}

void dilate(InputArray src, OutputArray dst, int itterations = 1) {
	int erosion_size = 1;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
		cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		cv::Point(erosion_size, erosion_size));
	dilate(src, dst, element, Point(-1, -1), itterations);
}

void erode(InputArray src, OutputArray dst, int itterations = 1) {
	int erosion_size = 1;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
		cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		cv::Point(erosion_size, erosion_size));
	erode(src, dst, element, Point(-1, -1), itterations);
}

void drawCircle(Mat dst, vector<Vec3f> circles) {
	for (size_t i = 0; i < circles.size(); i++) {
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// draw the circle center
		circle(dst, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		// draw the circle outline
		circle(dst, center, radius, Scalar(0, 0, 255), 3, 8, 0);
	}
}





void blobDetector(Mat src) {
	// Setup SimpleBlobDetector parameters.
	SimpleBlobDetector::Params params;

	// Change thresholds
	params.minThreshold = 50;
	params.maxThreshold = 500;

	// Filter by Area.
	params.filterByArea = true;
	params.minArea = 200;

	// Filter by Circularity
	params.filterByCircularity = false;
	params.minCircularity = .2;

	// Filter by Convexity
	params.filterByConvexity = false;
	params.minConvexity = 0.87;

	// Filter by Inertia
	params.filterByInertia = false;
	params.minInertiaRatio = 0.01;


	// Storage for blobs
	vector<KeyPoint> keypoints;


	// Set up detector with params
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

	// Detect blobs
	detector->detect(src, keypoints);

	// Draw detected blobs as red circles.
	// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures
	// the size of the circle corresponds to the size of blob

	Mat im_with_keypoints;
	drawKeypoints(src, keypoints, im_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	// Show blobs
	imshow("keypoints", im_with_keypoints);
}