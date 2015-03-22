/*
Copyright (c) 2015, Mostafa Mohamed (Izz)
izz.mostafa@gmail.com

All rights reserved.

Redistribution and use in source and binary forms, with or without modification
, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching/stitcher.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define radius 7
#define PI 3.14159265359
#define _VIS 1

using namespace std;
using namespace cv;

class Visualization
{
	// rotate point
	// orig: is the center point
	// inp : is the point to be rotated around the center
	// theta: is the angel in radians
	static Point2f rotate_point(Point2f orig,Point2f inp,double theta)
	{
		Point2f res;
		inp = inp - orig;
		// x_new = x cos theta - y sin theta
		res.x = inp.x * cos(theta) - inp.y * sin(theta);
		// y_new = x sin theta + y cos theta
		res.y = inp.x * sin(theta) + inp.y * cos(theta);
		// translate back
		return (res + orig);
	}
public:
	// draw SIFT key point
	static Mat drawKeyPoint(Mat input,KeyPoint kp,cv::Scalar color)
	{
		Mat res = input.clone();

		// calculate other points
		// we need 5 horizontal lines and 5 vertical lines
		// the lines needs 12 points as follows
		// the vertical lines
		for(double x = kp.pt.x - kp.size; x <= kp.pt.x + kp.size + 0.0001;x += (kp.size / 2))
		{
			// get the points
			Point2f p1 = rotate_point(kp.pt,Point2f(x,kp.pt.y - kp.size),kp.angle*PI/180);
			Point2f p2 = rotate_point(kp.pt,Point2f(x,kp.pt.y + kp.size),kp.angle*PI/180);
			
			// then draw the vertical lines
			cv::line(res,p1,p2,color);
		}
		// the horizontal lines
		for(double y = kp.pt.y - kp.size; y <= kp.pt.y + kp.size + 0.0001;y += (kp.size / 2))
		{
			// get the points and rotate them
			Point2f p1 = rotate_point(kp.pt,Point2f(kp.pt.x - kp.size,y),kp.angle*PI/180);
			Point2f p2 = rotate_point(kp.pt,Point2f(kp.pt.x + kp.size,y),kp.angle*PI/180);
			// draw the lines
			cv::line(res,p1,p2,color);
		}

		return res;
	}

	template <typename T>
	static Mat overlayFidsImage(Mat tmp,vector<T>& res,cv::Scalar color = cv::Scalar(255,255,255),bool text = false)
	{
		Mat img = tmp.clone();
		for(int j = 0;j < res.size();j++)
		{
			if(text)
			{
				char number[10];
				sprintf(number,"%d", j + 1);
				putText(img, number, cv::Point(res[j].x,res[j].y), 
					FONT_HERSHEY_COMPLEX_SMALL, 0.7, color, 1, CV_AA);
			}
			else
			{
				cv::circle(img,cv::Point(res[j].x,res[j].y),2,color,-2,CV_AA);
				//cv::circle(img,cv::Point(res[j].x,res[j].y),3,cv::Scalar(0,0,0),1);
			}
		}
		return img;
	}

	template <typename T>
	static Mat overlayFidsImage(Mat tmp,vector<T>& res,float rad,bool text = false)
	{
		Mat img = tmp.clone();
		for(int j = 0;j < res.size();j++)
		{
			if(text)
			{
				char number[10];
				sprintf(number,"%d", j + 1);
				putText(img, number, cv::Point(res[j].x,res[j].y), 
					FONT_HERSHEY_COMPLEX_SMALL, 0.4, cv::Scalar(0,0,0), 1, CV_AA);
			}
			else
			{
				cv::circle(img,cv::Point(res[j].x,res[j].y),rad,cv::Scalar(255,255,255),-2,CV_AA);
				cv::circle(img,cv::Point(res[j].x,res[j].y),rad,cv::Scalar(0,0,0),1,CV_AA);
			}
		}
		return img;
	}

	template<typename T>
	static void displayFidsImage(Mat tmp,vector<T>& res,cv::Scalar color = cv::Scalar(255,255,255),bool text = false)
	{
		//imshow("output", tmp);
		//cvWaitKey(0);
		Mat img = overlayFidsImage(tmp,res,color,text);
		imshow("output", img);
		cvWaitKey(0);
	}

	static Mat overlayDetectedFidsImage(Mat tmp,vector<DetecetedPoint>& res,bool text = false)
	{
		Mat img = tmp.clone();
		for(int j = 0;j < res.size();j++)
		{
			if(text)
			{
				char number[50];
				sprintf(number,"%0.2llf", res[j].prob);
				putText(img, number, cv::Point(res[j].x,res[j].y), 
					FONT_HERSHEY_COMPLEX_SMALL, 0.4, cvScalar(0,255,0), 1, CV_AA);
			}
			else
			{
				cv::circle(img,cv::Point(res[j].x,res[j].y),2,cv::Scalar(255,255,255),-2);
				cv::circle(img,cv::Point(res[j].x,res[j].y),3,cv::Scalar(0,0,0),1);
			}
		}
		return img;
	}

	static void displayDetectedFidsImage(Mat tmp,vector<DetecetedPoint>& res,bool text = false)
	{
		Mat img = overlayDetectedFidsImage(tmp,res,text);
		imshow("output", img);
		cvWaitKey(0);
	}
};

#endif
