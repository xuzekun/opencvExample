#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::face;

#define COLOR Scalar(255,200,0)

void drawPolyline
(
	Mat &im,
	const vector<Point2f> &landmarks,
	const int start,
	const int end,
	bool isClosed = false
)
{
	// Gather all points between the start and end indices
	vector <Point> points;
	for (int i = start; i <= end; i++)
	{
		points.push_back(cv::Point(landmarks[i].x, landmarks[i].y));
	}
	// Draw polylines. 
	polylines(im, points, isClosed, COLOR, 2, 16);
}

void drawLandmarks(Mat &im, vector<Point2f> &landmarks)
{
	// Draw face for the 68-point model
	if (landmarks.size() == 68)
	{
		drawPolyline(im, landmarks, 0, 16);           // Jaw line
		drawPolyline(im, landmarks, 17, 21);          // Left eyebrow
		drawPolyline(im, landmarks, 22, 26);          // Right eyebrow
		drawPolyline(im, landmarks, 27, 30);          // Nose bridge
		drawPolyline(im, landmarks, 30, 35, true);    // Lower nose
		drawPolyline(im, landmarks, 36, 41, true);    // Left eye
		drawPolyline(im, landmarks, 42, 47, true);    // Right Eye
		drawPolyline(im, landmarks, 48, 59, true);    // Outer lip
		drawPolyline(im, landmarks, 60, 67, true);    // Inner lip
	}
	else
	{ // If the number of points is not 68, we do not know which 
	  // points correspond to which facial features. So, we draw 
	  // one dot per landamrk. 
		for (int i = 0; i < landmarks.size(); i++)
		{
			circle(im, landmarks[i], 3, COLOR, FILLED);
		}
	}
}

void writeFile(vector<Point2f>& landmarks)
{
	ofstream file;
	file.open("face.txt");
	for (int i = 0; i < landmarks.size(); i++)
	{
		file << landmarks[i].x <<" "<< landmarks[i].y<<endl;
	}
}

void readFile()
{
	ifstream ifs("face.txt");
	float x, y;
	while (ifs >> x >> y)
	{
		cout << x << " " << y << endl;
	}
}


int main()
{
	CascadeClassifier faceDetector("haarcascade_frontalface_alt2.xml");
	Ptr<Facemark> facemark = FacemarkLBF::create();
	facemark->loadModel("lbfmodel.yaml");

	Mat srcImg = imread("5.jpg");

	Mat grayImg;
	cvtColor(srcImg, grayImg, COLOR_BGR2GRAY);

	vector<Rect> faces;

	faceDetector.detectMultiScale(grayImg, faces);

	vector<vector<Point2f>> landmarks;



	bool success = facemark->fit(grayImg, faces, landmarks);
	if (success)
	{
		for (int i = 0; i < landmarks.size(); i++)
		{
			drawLandmarks(srcImg, landmarks[i]);
			writeFile(landmarks[i]);
		}
	}
	readFile();
	imshow("", srcImg);
	waitKey();
}