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




//OoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoO
void findTarget(Mat &frame) {
    Mat targetMask, targetOutline, targetLines, targetContours;
    vector<vector<Point> > contours;
    vector<Vec2f> lines;
    
    inRangeHSV(frame, targetMask, Scalar(85, 180, 238), Scalar(125, 255, 255), true);
    dilate(targetMask, targetMask, 3);
    dilate_erode(targetMask, targetMask, 2);
    detectContours(targetMask, targetContours, contours, 155);
    
    vector<vector<Point2f> > boxes;
    vector<Point2f> targetCenters;
    vector<Size> targetDimensions;
    Point2f largestTargetCenter;
    Size largestTargetDimensions;
    //LOCATE TARGETS AND SELECT LARGEST ONE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    int largestTargetArea = 0;
    for (int i = 0; i < contours.size(); ++i) {
        Rect contourBounds = boundingRect(contours[i]);
        if (300 <= contourBounds.size().area()) {
            Point center = Point(contourBounds.x + contourBounds.width / 2, contourBounds.y + contourBounds.height / 2);
            targetCenters.push_back(center);
            
            if (contourBounds.size().area() > largestTargetArea) {
                largestTargetCenter = center;
                largestTargetDimensions = contourBounds.size();
                largestTargetArea = contourBounds.area();
            }
            targetDimensions.push_back(contourBounds.size());
        }
    }
    //DRAW RECTANGLE AROUND TARGET - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    rectangle(frame, Rect(largestTargetCenter.x - largestTargetDimensions.width/2, largestTargetCenter.y - largestTargetDimensions.height/2, largestTargetDimensions.width, largestTargetDimensions.height), Scalar(0, 255, 0), 1);
    rectangle(frame, Point(largestTargetCenter.x - 2, largestTargetCenter.y - 2), Point(largestTargetCenter.x + 2, largestTargetCenter.y + 2), Scalar(255, 255, 0), 5);
    //SOLVING DISTANCE AND ANGLE DIFFERENTIAL - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    double targetX = largestTargetCenter.x; //pixels
    double targetY = largestTargetCenter.y;
    double imageX = frame.size().width;
    double imageY = frame.size().height;
    const double VIEW_ANGLE_X = 67; //degrees
    const double VIEW_ANGLE_Y = 51;
    const double CAMERA_ANGLE = 57; /*from ground, not from normal*/
    const double CAMERA_HEIGHT = 14; //inches
    const double TARGET_HEIGHT = 90;
    
    double PixelDistance = pow(pow(imageX/2 - targetX, 2.0) + pow(imageX/(2*tan(M_PI*VIEW_ANGLE_X/360)),2),.5);
    double PixelHeight = PixelDistance*tan(M_PI*(CAMERA_ANGLE - VIEW_ANGLE_Y/2)/180) + imageY - targetY;
    double TargetDistance = PixelDistance*(TARGET_HEIGHT - CAMERA_HEIGHT)/PixelHeight; //inches
    
    double AngleDifferential = VIEW_ANGLE_X*targetX/imageX - VIEW_ANGLE_X/2;
    //DISPLAYING DISTANCE AND ANGLE DIFFERENTIAL - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    int textLine = 1;
    string text = string("Target Distance: ") + to_string((int)round(TargetDistance));
    putText(frame, text, Point(4, 12 * textLine + 4), CV_FONT_NORMAL, .4, Scalar(255, 255, 255));
    ++textLine;
    string text1 = string("Angle Differential: ") + to_string((int)round(AngleDifferential));
    putText(frame, text1, Point(4, 12 * textLine + 4), CV_FONT_NORMAL, .4, Scalar(255, 255, 255));
    ++textLine; ++textLine;
    string text2 = string("Center: ") + to_string((int)round(largestTargetCenter.x)) + ", " + to_string((int)round(largestTargetCenter.y));
    putText(frame, text2, Point(4, 12 * textLine + 4), CV_FONT_NORMAL, .4, Scalar(255, 255, 255));
    ++textLine; ++textLine; ++textLine;
    //WRITING INFORMATION TO NETWORK TABLES
    double TARGET_OFFSET_X = 15; //28 for competition, 15 for practice
    if (largestTargetArea > 0) {
        table->PutBoolean("TargetVisibility", true);
        table->PutNumber("TargetX", largestTargetCenter.x - TARGET_OFFSET_X);
        table->PutNumber("TargetY", largestTargetCenter.y);
        table->PutNumber("TargetWidth", largestTargetDimensions.width);
        table->PutNumber("TargetHeight", largestTargetDimensions.height);
        table->PutNumber("ImageX", frame.size().width);
        table->PutNumber("ImageY", frame.size().height);
        table->PutNumber("TargetDistance", TargetDistance);
        table->PutNumber("AngleDifferential", AngleDifferential);
    }else {
        table->PutBoolean("TargetVisibility", false);
    }
    //OPEN INTERFACE WINDOW - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    imwrite("C:\\Users\\Ben\\Documents\\!Dev\\FRC\\frame.jpeg", frame);
    imshow("in", frame);
}
//OoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoO




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
}

int main() {
    initNetworkTables();
    runFromRobot("http://10.42.56.20/mjpg/video.mjpg");
    return 0;
}