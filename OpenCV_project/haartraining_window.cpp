#include "opencv2/opencv.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <conio.h>
#include <fstream>
#include <windows.h>

#include <string>
#include <sstream>
#include <vector>

using namespace cv;
using namespace std;

const string windowName = "PositiveWindowBuilder";
const int MIN_X_LOCATION = 2;
const int MAX_X_LOCATION = 3;
const int MIN_Y_LOCATION = 4;
const int MAX_Y_LOCATION = 5;

Mat frame;
int frameIndex = 0;
int frameCount;
//string* posLines;

char posFilePath[256]; // <- danger, only storage for 256 characters.
vector<vector<string>> posLineData;
//ofstream posFileOut;

boolean selecting;
int startX = -1;
int startY = -1;
int endX = -1;
int endY = -1;

std::vector<std::string> &split(std::string &s, char delim, std::vector<std::string> &elems) {
	std::stringstream ss(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		elems.push_back(item);
	}
	return elems;
}


std::vector<std::string> split(std::string &s, char delim) {
	std::vector<std::string> elems;
	split(s, delim, elems);
	return elems;
}

string addChars(const char *args, ...) {
	string result = "";

	va_list arguments;
	for (va_start(arguments, args); args != NULL; args = va_arg(arguments, const char *)) {
		result += args;
		//cout << args << endl;
	}
	va_end(arguments);

	return result;
}

void a(string in) {
	cout << in << endl;
}

// dirName must end with a pathname separator
void writeDirectoryContentsToFile(const string dirName, const string& imageSubDirName)
{
	string fullDirPath = dirName + imageSubDirName;
	string output;
	WIN32_FIND_DATAA findData;
	HANDLE searchHandle = FindFirstFileA((fullDirPath + "\\*").c_str(), &findData);
	bool foundFiles = (searchHandle != NULL);
	while (foundFiles)
	{
		if ((findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0)
			output += fullDirPath + "\\" + findData.cFileName + "\n";
		foundFiles = (bool) FindNextFileA(searchHandle, &findData);
	}

	//Construct file
	ofstream newFile(dirName + imageSubDirName+".txt");
	newFile << output;
}

void savePosFile() {
	ofstream posFileOut(posFilePath);

	string line = "";
	for (int i = 0; i < frameCount; ++i) {
		line += posLineData.at(i).at(0);
		line += " 1";
		for (int j = 2; j <= 5; ++j) {
			line += " " + posLineData.at(i).at(j);
		}

		posFileOut << line + "\n";
		//cout << line << endl;
		line = "";
	}
}

void readPos(const char* dir) {
	//Get full filepath
	strncpy(posFilePath, dir, sizeof(posFilePath));
	strncat(posFilePath, "negative.txt", sizeof(posFilePath));

	{
		//Get size of positive file
		ifstream posFile(posFilePath);
		frameCount = 0;
		std::string line;
		//posLines = new string[frameCount]; //Size set

		//Read positive file to array
		//posFile.clear();
		//posFile.seekg(0, ios::beg);

		while (std::getline(posFile, line)) {
			posLineData.push_back(split(line, ' '));
			posLineData.at(frameCount).resize(6);
			++frameCount;
		}
	}

	//Init output stream
	savePosFile();
}

void drawSelection(double eX, double eY) {
	cout << "pos " << startX << "," << startY << "; " << eX << "," << eY << endl;
	Mat newFrame;
	frame.copyTo(newFrame);
	rectangle(newFrame, Point(startX, startY), Point(eX, eY), Scalar(0, 0, 255));
	imshow(windowName, newFrame);
}

void loadImage() {
	//string currentLine(posLineData.at(frameIndex).begin(), posLineData.at(frameIndex).end());//Convert to string
	//cout << "Loading " << currentLine << endl;//Print to console
	 
	//cout << endl << "Loading" << posLineData.at(frameIndex).at(0) << endl;
	frame = imread(posLineData.at(frameIndex).at(0), CV_LOAD_IMAGE_COLOR);
	imshow(windowName, frame);

	try {
		startX = atoi(posLineData.at(frameIndex).at(MIN_X_LOCATION).c_str());
		endX = atoi(posLineData.at(frameIndex).at(MAX_X_LOCATION).c_str());
		startY = atoi(posLineData.at(frameIndex).at(MIN_Y_LOCATION).c_str());
		endY = atoi(posLineData.at(frameIndex).at(MAX_Y_LOCATION).c_str());
		drawSelection(endX, endY);
	}catch (...) {}
}

void onWindowMouseEvent(int event, int x, int y, int flags, void* userdata) {
	//Detect posivitve image selection input
	if (event == EVENT_LBUTTONDOWN) {
		selecting = true;
		startX = x;
		startY = y;
	}else if (event == EVENT_LBUTTONUP) {
		selecting = false;
		if (startX != x && startY != y) {
			endX = x;
			endY = y;

			//Process positive image location
			posLineData.at(frameIndex).at(MIN_X_LOCATION) = to_string(min(startX, endX));//minX
			posLineData.at(frameIndex).at(MAX_X_LOCATION) = to_string(max(startX, endX));//maxX
			posLineData.at(frameIndex).at(MIN_Y_LOCATION) = to_string(min(startY, endY));//minY
			posLineData.at(frameIndex).at(MAX_Y_LOCATION) = to_string(max(startY, endY));//MaxY

			//Save positive image location to file
			savePosFile();
		}
	}else if (event == EVENT_MOUSEMOVE && selecting) {
		drawSelection(x, y);
	}
}

void startHaarTrainingWindow(const char* dir) {
	//Read positive file
	readPos(dir);
	writeDirectoryContentsToFile(dir, "negative");
	writeDirectoryContentsToFile(dir, "positive");

	//Init window
	namedWindow(windowName);
	loadImage();
	setMouseCallback(windowName, onWindowMouseEvent, NULL);

	//Key input
	while (true) {
		int key = waitKey(0);
		cout << key << endl;

		if (key == 2424832) {//Left 37
			if(0 < frameIndex) --frameIndex;
			loadImage();
		}else if (key == 2555904) {//Right 39
			if(frameIndex < frameCount) ++frameIndex;
			loadImage();
		}else if (key == 27) {//Esc
			break;//Terminate
		}
	}
}