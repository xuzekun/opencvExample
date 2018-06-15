#include <opencv2/opencv.hpp>
using namespace cv;

int main()
{
	CascadeClassifier faceCascade;
	faceCascade.load("haarcascade_frontalface_alt2.xml");

	vector<Rect> faces;
	VideoCapture capture("E:\\\TDDownload\\\TEAM-089.avi");
	Mat srcImg;
	Mat grayImg;
	while (1)
	{
		capture >> srcImg;
		cvtColor(srcImg, grayImg, COLOR_RGB2GRAY);

		faceCascade.detectMultiScale(grayImg, faces, 1.1, 6, 0, Size(0, 0));
		if (faces.size() > 0)
		{
			for (int i = 0; i < faces.size(); i++)
			{
				rectangle(srcImg, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(0, 255, 0), 1, 8);

			}
		}
		imshow("1.jpg", srcImg);
		waitKey(10);
	}
	waitKey(0);
	return 0;
}
