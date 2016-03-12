//OpenCV
#include "opencv2/opencv.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//FRC
#include "networktables\NetworkTable.h"

#include <iostream>
#include <stdio.h>
#include <chrono>
#include <thread>

#include "tools.h"
#include "haartraining_window.h"

#include <math.h>

using namespace cv;
using namespace std;

# define M_PI           3.14159265358979323846  /* pi */

std::shared_ptr<NetworkTable> table;

vector<cv::Rect> detectedOranges;
CascadeClassifier orangeCascade;
void init() {
	orangeCascade.load("C:\\Users\\Ben\\Desktop\\HaarTraining\\tooth\\data\\cascade\\cascade.xml");
}

/*void cascadeDetection(Mat frame) {
	orangeCascade.detectMultiScale(frame, detectedOranges, 1.1, 5, 0, Size(10, 10));//Harr
	//orangeCascade.detectMultiScale(frame, detectedOranges, 1.1, 20);//LBP

	//Draw all matches
	for (size_t i = 0; i < detectedOranges.size(); i++) {
		cv::Point center(detectedOranges[i].x + detectedOranges[i].width*0.5, detectedOranges[i].y + detectedOranges[i].height*0.5);
		ellipse(frame, center, Size(detectedOranges[i].width*0.5, detectedOranges[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
	}

	detectedOranges.clear();
	imshow("feed", frame);
}

void faultyBallDetection(Mat frame) {
	//imshow("input", frame);
	Mat white, gray, out, edges;
	//inRange(frame, Scalar(235, 235, 235), Scalar(255, 255, 255), white);
	inRangeRGB(frame, white, Scalar(100, 150, 150), Scalar(255, 255, 255), true); //Get white
	inRangeHSV(frame, gray, Scalar(20, 0, 0), Scalar(180, 100, 150), false); //Get gray
	dilate_erode(gray, gray, 10);
	//dilate(white, white, 2);

	Canny(frame, edges, 25, 60);
	//bitwise_and(white, edges, white);//gets instersection
	//bitwise_or(white, edges);
	//subtract(edges, white, edges);//subtracts white edges from edges
	dilate(edges, edges, 10);
	erode(edges, edges, 10);

	GaussianBlur(gray, gray, Size(9, 9), 2, 2);
	//dilate_erode(frame, frame, 10);
	addWeighted(gray, 10, frame, 1, 1, frame);
	//blobDetector(frame);
	//blobDetector(gray);

	imshow("final", frame);

	//imshow("white", white);
	//imshow("grey", gray);
	//imshow("feed", edges);
}

void oldBallDetect(Mat frame) {
	Mat gray, mask, out;

	//frame = resize(frame);
	//inRangeHSV(frame, mask, Scalar(20, 0, 0), Scalar(180, 100, 150), true);
	//91 149 255
	inRangeHSV(frame, mask, Scalar(20, 0, 0), Scalar(91, 149, 255), true);

	//Dilate mask to eliminate holes
	dilate_erode(mask, mask, 10);

	cvtColor(frame, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, gray, Size(9, 9), 2, 2);

	//Detect circles in the image
	vector<Vec3f> circles;
	HoughCircles(mask, circles, CV_HOUGH_GRADIENT, 2, mask.rows / 4, 30, 50, 10);

	//Draw circles
	drawCircle(frame, circles);

	//Show image
	//frame.copyTo(out, mask);//Apply mask
	imshow("feed", frame);
	//imshow("out", out);
}
/*
void processFeed(Mat frame) {
	//inRangeRGB(frame, mask, Scalar(20, 0, 0), Scalar(180, 100, 150), true);
	//	inRangeHSV(frame, mask, Scalar(20, 0, 0), Scalar(180, 100, 150), true);
	imshow("output", frame);
}*/


void detectLines(InputArray src, InputOutputArray dst, vector<Vec2f> lines222) {
	cvtColor(src, dst, CV_GRAY2BGR);

#if 0
	vector<Vec2f> lines;
	HoughLines(src, lines, 1, CV_PI / 180, 50, 0, 0);

	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(dst, pt1, pt2, Scalar(0, 0, 255), 1, CV_AA);
	}
#else
	vector<Vec4i> lines;
	HoughLinesP(src, lines, 1, CV_PI / 180, 20, 50, /*min length= */40);
	for (size_t i = 0; i < lines.size(); i++) {
		Vec4i l = lines[i];
		line(dst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 1, CV_AA);
	}
#endif
}


RNG rng(12345);
void detectContours(InputArray src, InputOutputArray dst, OutputArray contours, int contourMinimumLength) {
	Mat outline;
	//vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// Detect edges using canny
	Canny(src, outline, 50, 200, 3);

	/// Find contours
	findContours(outline, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Filter contours
	/*Mat tempContours = Mat::zeros(outline.size(), CV_8UC3);
	int i = 0;
	for (vector<vector<Point> >::iterator it = contours.begin(); it != contours.end(); ) {
		//if (it->size() < contourMinimumLength) {
		if (contourArea(contours[i]) < contourMinimumLength) {
			it = contours.erase(it);
		} else {
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			drawContours(tempContours, contours, i, color, 2, 8, hierarchy, 0, Point());
			++it; ++i;
		}
	}*/

	/// Draw contours
	Mat tempContours = Mat::zeros(outline.size(), CV_8UC3);
	for (int i = 0; i < hierarchy.size(); i++) {
		//double area = contourArea(contours[i]);
		//contours.erase(contours[i]);
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(tempContours, contours, i, color, 2, 8, hierarchy, 0, Point());
	}

	tempContours.copyTo(dst);
}

void findTarget(Mat &frame) {
	//resize(frame, frame, frame.size() / 2);
	Mat targetMask, targetOutline, targetLines, targetContours;
	vector<vector<Point> > contours;
	vector<Vec2f> lines;

	inRangeHSV(frame, targetMask, Scalar(85, 180, 238), Scalar(125, 255, 255), true);
	dilate(targetMask, targetMask, 3);
	dilate_erode(targetMask, targetMask, 2);

	//Canny(targetMask, targetOutline, 50, 200, 3);
	//detectLines(targetOutline, targetLines, lines);

	detectContours(targetMask, targetContours, contours, 155);
	vector<vector<Point2f> > boxes;
	vector<Point2f> targetCenters;
	vector<Size> targetDimensions;

	Point2f largestTargetCenter;
	Size largestTargetDimensions;
	int largestTargetArea = 0;

	for (int i = 0; i < contours.size(); ++i) {
		Rect contourBounds = boundingRect(contours[i]);
	//	rectangle(frame, contourBounds, Scalar(0, 255, 0), 1);

		if (300 <= contourBounds.size().area()) {
			///Draw center point
			Point center = Point(contourBounds.x + contourBounds.width / 2, contourBounds.y + contourBounds.height / 2);
		//	rectangle(frame, Point(center.x - 2, center.y - 2), Point(center.x + 2, center.y + 2), Scalar(255, 255, 0), 5);
			targetCenters.push_back(center);

			//Get largest target
			if (contourBounds.size().area() > largestTargetArea) {
				largestTargetCenter = center;
				largestTargetDimensions = contourBounds.size();
				//largestTargetArea = contourBounds.size().area();
				largestTargetArea = contourBounds.area();
			}

			///Get target size
			targetDimensions.push_back(contourBounds.size());
		}
	}
	rectangle(frame, Rect(largestTargetCenter.x - largestTargetDimensions.width/2, largestTargetCenter.y - largestTargetDimensions.height/2, largestTargetDimensions.width, largestTargetDimensions.height), Scalar(0, 255, 0), 1);
	rectangle(frame, Point(largestTargetCenter.x - 2, largestTargetCenter.y - 2), Point(largestTargetCenter.x + 2, largestTargetCenter.y + 2), Scalar(255, 255, 0), 5);
	///Display info
	int textLine = 1;

	//write info
	string text = string("Center: ") + to_string((int)round(largestTargetCenter.x)) + ", " + to_string((int)round(largestTargetCenter.y));
	putText(frame, text, Point(4, 12 * textLine + 4), CV_FONT_NORMAL, .4, Scalar(255, 255, 255));
	++textLine;

	string text2 = string("Dimensions: ") + to_string((int)round(largestTargetDimensions.width)) + ", " + to_string((int)round(largestTargetDimensions.height));
	putText(frame, text2, Point(4, 12 * textLine + 4), CV_FONT_NORMAL, .4, Scalar(255, 255, 255));
	++textLine; ++textLine;
	//DISTANCE FINDING EQUATIONS--------------------------------------------------------------------------------------------------------------
		double imageX = frame.size().width;
		double imageY = frame.size().height;
		double targetX = largestTargetCenter.x;
		double targetY = largestTargetCenter.y;
		const double VIEW_ANGLE_X = 67; //degrees
		const double VIEW_ANGLE_Y = 51; //degrees
		const double CAMERA_ANGLE = 33; //degrees
		const double CAMERA_HEIGHT = 13.5; //inches
		const double TARGET_HEIGHT = 90; //inches
		//write haydenDistance and haydenAngleAwayFromTarget
		double angleDifferential = tan((((imageX / 2) - targetX) / ((imageX / 2) / tan((VIEW_ANGLE_X / 2)*(M_PI / 180))))*(M_PI / 180));
		double pixelDistance = ((imageX / 2) - targetX)*((imageX / 2) - targetX)
			+ ((imageX / 2) / tan((VIEW_ANGLE_X / 2)*(M_PI / 180)))*((imageX / 2) / tan((VIEW_ANGLE_X / 2)*(M_PI / 180)));
		if (pixelDistance < 0) {
			pixelDistance = -pixelDistance;
		}
		pixelDistance = sqrt(pixelDistance);
		double pixelHeight = pixelDistance * tan((VIEW_ANGLE_Y + CAMERA_ANGLE) * (M_PI / 180));
		double inchesDistanceHAYDEN = (pixelDistance * (TARGET_HEIGHT - CAMERA_HEIGHT)) / (pixelHeight - targetY);
		//write openSourceDistance
		targetY = -((2 * (targetY / frame.size().height)) - 1);
		double inchesDistanceOPENSOURCE = (TARGET_HEIGHT - CAMERA_HEIGHT) /
			tan((targetY * VIEW_ANGLE_Y / 2.0 + CAMERA_ANGLE) * M_PI / 180);
	
	string text3 = string("Target Distance (Hayden's rough draft): ") + to_string(round(inchesDistanceHAYDEN));
	putText(frame, text3, Point(4, 12 * textLine + 4), CV_FONT_NORMAL, .4, Scalar(255, 255, 255));
	++textLine, ++textLine, ++textLine;

	string text4 = string("Target Distance (Open source - unreliable?): ") + to_string(round(inchesDistanceOPENSOURCE));
	putText(frame, text4, Point(4, 12 * textLine + 4), CV_FONT_NORMAL, .4, Scalar(255, 255, 255));
	++textLine, ++textLine, ++textLine; ++textLine;


	///Write to Network Table
	if (largestTargetArea > 0) {
		table->PutBoolean("TargetVisibility", true);
		table->PutNumber("TargetX", largestTargetCenter.x);
		table->PutNumber("TargetY", largestTargetCenter.y);
//		table->PutNumber("TargetWidth", largestTargetDimensions.width);
//		table->PutNumber("TargetHeight", largestTargetDimensions.height);
//		table->PutNumber("ImageWidth", frame.size().width);
//		table->PutNumber("ImageHeight", frame.size().height);
		table->PutNumber("AngleDifferential", angleDifferential);
		table->PutNumber("HaydenTargetDistance", inchesDistanceHAYDEN);
		table->PutNumber("DistanceToTarget", inchesDistanceOPENSOURCE);
	}
	else {
		table->PutBoolean("TargetVisibility", false);
	}

	imwrite("C:\\Users\\Ben\\Documents\\!Dev\\FRC\\frame.jpeg", frame);
	imshow("in", frame);
	//imshow("mask", targetMask);
	//imshow("outline", targetOutline);
//	imshow("lines", targetLines);
	//imshow("contours", targetContours);

}

void processFeed(Mat &frame) {
	//bioDetect(frame);
	//cascadeDetection(frame);
	//faultyBallDetection(frame);
	findTarget(frame);
	//imshow("testFrame", frame);
}


void runFromStream(VideoCapture &stream1) {
	if (!stream1.isOpened()) { //check if video device / video file has been initialised
		cout << "cannot open video source";
		return;
	}

	//unconditional loop
	while (true) {
		Mat cameraFrame;
		if (stream1.read(cameraFrame))
			processFeed(cameraFrame);//proccess
		//std::this_thread::sleep_for(std::chrono::milliseconds(100));//slow video
		if (waitKey(30) >= 0)//wait for esc key
			break;
	}

}

void runFromRobot(string ip) {
	VideoCapture stream1(ip);
	//runFromStream(stream1);


	int ex = static_cast<int>(stream1.get(CV_CAP_PROP_FOURCC));
	Size frameSize = Size((int)stream1.get(CV_CAP_PROP_FRAME_WIDTH), (int)stream1.get(CV_CAP_PROP_FRAME_HEIGHT));

	VideoWriter videoWriter;
	String videoFileName = "recording";
	videoWriter.open("C:\\Users\\FRC1\\Pictures\\FRC Recordings\\" + videoFileName + ".avi", ex, stream1.get(CV_CAP_PROP_FPS), frameSize, true);

	if (!stream1.isOpened()) { //check if video device / video file has been initialised
		cout << "cannot open video source";
		return;
	}

	//unconditional loop
	while (true) {
		Mat cameraFrame;
		if (stream1.read(cameraFrame)) {
			videoWriter << cameraFrame;
			processFeed(cameraFrame);//proccess
		}
		//std::this_thread::sleep_for(std::chrono::milliseconds(100));//slow video
		if (waitKey(30) >= 0)//wait for esc key
			break;
	}
}

void runFromWebcam() {
	VideoCapture stream1(0);//0 is the id of video device.0 if you have only one camera.
	runFromStream(stream1);
}

void runFromVideo(const char* filename) {
	VideoCapture stream1(filename);
	runFromStream(stream1);
}


void runFromImage(const char* filename) {
	Mat image;
	image = imread(filename, CV_LOAD_IMAGE_COLOR);
	processFeed(image);//proccess
	waitKey(0);//waits for keypress to break
}

//std::shared_ptr<NetworkTable> table;
void initNetworkTables() {
	
	NetworkTable::SetClientMode();
	//NetworkTable::SetIPAddress("10.42.56.21");//practice robot IP
	NetworkTable::SetIPAddress("10.42.56.2");//test board IP
	//NetworkTable::SetTeam(4256);
	//NetworkTable::Initialize();
	table = NetworkTable::GetTable("SaltVision");

	//table->PutNumber("TargetCenterX", 0);
}
 
int main() {
	initNetworkTables();
	//runFromImage("C:\\Users\\Ben\\Downloads\\Shareit\\Photo\\bio-gel\\IMG_6045.JPG");

	//init();
	runFromRobot("http://10.42.56.20/mjpg/video.mjpg");//rtsp://FRC:FRC@10.42.56.20/axis-media/media.amp
	//runFromImage("C:\\Users\\frc1\\workspace\\opencv_frc_2016 - Copy\\input.png");

	//runFromRobot("rtsp://frc:frc@10.42.56.20/axis-media/media.amp");
	//runFromWebcam();
	//runFromVideo("C:\\Users\\Ben\\Downloads\\Shareit\\Video\\IMG_6005.mov");
	//runFromImage("C:\\Users\\Ben\\Downloads\\Shareit\\Video\\ball\\o_19a557dafdb6ed65-41.jpg");
	//runFromImage("C:\\Users\\Ben\\Downloads\\Shareit\\Video\\ball\\o_19a557dafdb6ed65-35.jpg");
	//runFromImage("C:\\Users\\Ben\\Downloads\\Shareit\\Video\\ball\\o_19a557dafdb6ed65-67.jpg");
	//runFromImage("C:\\Users\\Ben\\Downloads\\Shareit\\Video\\ball\\o_19a557dafdb6ed65-49.jpg");

	//runFromVideo("C:\\Users\\Ben\\Downloads\\Shareit\\Video\\IMG_5957_converted.mov");
	//runFromVideo("C:\\Users\\Ben\\Downloads\\Shareit\\Video\\orange.mov");
	//runFromImage("C:\\Users\\Ben\\Desktop\\HaarTraining\\tooth\\positive\\o_fdec6778b00739a2-0.bmp");
	//cerr << cv::getBuildInformation();
	//startHaarTrainingWindow("C:\\Users\\Ben\\Desktop\\HaarTraining\\tooth\\");
	//std::this_thread::sleep_for(std::chrono::milliseconds(1000));//pause
	return 0;
}