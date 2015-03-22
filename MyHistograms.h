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


#ifndef MYHISTOGRAMS_H
#define MYHISTOGRAMS_H

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

using namespace cv;
/*
Mat getHistofLBP(Mat &img, float minx, float miny,float maxx, float maxy, int binSize, bool normalize = true)
{
	Mat temp,hist = Mat::zeros(1, 256 / binSize, CV_32F);
	try
	{
		temp = lbp::histogram(img(cv::Rect(minx,miny,maxx - minx,maxy - miny)), 256);
		float sum = 0;
		for(int i = 0;i < 256;i++)
		{
			hist.at<float>(cv::Point(i/binSize, 0)) += temp.at<int>(cv::Point(i, 0));
			if(normalize)
			{
				sum += temp.at<int>(cv::Point(i, 0));
			}
		}
		// normalize
		if(normalize)
		{
			for(int i = 0;i < hist.cols;i++)
			{
				hist.at<float>(cv::Point(i, 0)) /= sum;
			}
		}
	}
	catch(...)
	{
		printf("img.cols = %d minx = %d maxx = %d\n", img.cols, minx, maxx);
		printf("img.rows = %d miny = %d maxy = %d\n", img.rows, miny, maxy);
	}
	return hist;
}
*/

/*
Mat getQuadHist(KeyPoint& k, Mat &dst,int binSize = 8)
{
	float minx = max(0.0f,k.pt.x - k.size);
	float miny = max(0.0f,k.pt.y - k.size);
	float maxx = min(k.size * 2 + minx, (float)dst.cols - 1);
	float maxy = min(k.size * 2 + miny, (float)dst.rows - 1);
	Mat hist = getHistofLBP(dst,minx,miny,k.pt.x, k.pt.y,binSize);
	hconcat(hist,getHistofLBP(dst,minx,k.pt.y,k.pt.x, maxy,binSize),hist);
	hconcat(hist,getHistofLBP(dst,k.pt.x,miny,maxx, k.pt.y,binSize),hist);
	hconcat(hist,getHistofLBP(dst,k.pt.x,k.pt.y,maxx, maxy,binSize),hist);

	double sum = 0;
	float* tmp = (float*)hist.data;
	for(int i = 0;i < hist.cols;i++)
	{
		sum += tmp[i];
	}
	for(int i = 0;i < hist.cols;i++)
	{
		tmp[i] /= sum;
	}
	return hist;
}
*/

/*
Mat getQuadQuadHist(KeyPoint& k, Mat &dst,int binSize = 32)
{
	Mat hist;
	for(int dx = -1;dx <= 1;dx+=2)
	{
		for(int dy = -1;dy <= 1;dy+=2)
		{
			KeyPoint newK = k;
			newK.pt.x = k.pt.x + dx * k.size / 2;
			newK.pt.y = k.pt.y + dy * k.size / 2;
			newK.size = k.size / 2;

			Mat tmp = getQuadHist(k,dst, binSize);
			if(hist.cols == 0)
			{
				hist = tmp;
			}
			else
			{
				hconcat(hist,tmp,hist);
			}
		}
	}

	return hist;
}
*/

/*
Mat getHistofLBP(Mat &img, vector<KeyPoint>& keypoints)
{
	// convert to gray
	Mat gray;
	cvtColor( img, gray, CV_RGB2GRAY );
	Mat lbp;
	// get bp
	lbp::OLBP(gray, lbp);
	normalize(lbp, lbp, 0, 255, NORM_MINMAX, CV_8UC1);
	// calculate histograms
	Mat desc;
	for(int i = 0;i < keypoints.size();i++)
	{
		KeyPoint k = keypoints[i];
		Mat rot_mat = getRotationMatrix2D(k.pt,k.angle,1);
		/// Rotate the lbp image
		Mat dst;
		warpAffine( lbp, dst, rot_mat, lbp.size() );
		
		Mat hist = getQuadQuadHist(k,dst);
		k.size /= 2;
		//
		hconcat(hist,getQuadQuadHist(k,dst),hist);

		if(desc.cols == 0)
		{
			desc = hist.clone();
		}
		else
		{
			vconcat(desc,hist,desc);
		}
	}
	return desc;
}
*/
/*
void getMinMax(Mat &m)
{
	//Initialize m
	double minVal; 
	double maxVal; 
	Point minLoc; 
	Point maxLoc;

	minMaxLoc( m, &minVal, &maxVal, &minLoc, &maxLoc );

	cout << "min val : " << minVal << endl;
	cout << "max val: " << maxVal << endl;
}
*/
/*
Mat getGradientSobel(Mat src)
{
	Mat src_gray;
	Mat grad;
	//char* window_name = "Sobel Demo - Simple Edge Detector";
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	int c;

  
	GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

	/// Convert it to gray
	cvtColor( src, src_gray, CV_RGB2GRAY );

  
	/// Generate grad_x and grad_y
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	//getMinMax(src);
	//getMinMax(src_gray);
	/// Gradient X
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	grad_x.convertTo(abs_grad_x, CV_32FC1);
	
	/// Gradient Y
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	grad_y.convertTo(abs_grad_y, CV_32FC1);
	
	grad = abs_grad_x.clone();

	// then calculate the angels
	int sz = abs_grad_x.rows * abs_grad_x.cols;
	for(int i = 0;i < sz;i++)
	{
		((float*)grad.data)[i] = atan2(((float*)abs_grad_y.data)[i], ((float*)abs_grad_x.data)[i]) * 127 / 3.14159 + 127;
	}

	//getMinMax(grad);

	grad.convertTo(grad, CV_8UC1);

	return grad;
}
*/
/*
Mat getHistofGradient(Mat &img, vector<KeyPoint>& keypoints)
{
	// calculate histograms
	Mat desc;
	for(int i = 0;i < keypoints.size();i++)
	{
		KeyPoint k = keypoints[i];
		Mat rot_mat = getRotationMatrix2D(k.pt,k.angle,1);
		/// Rotate the image
		Mat dst;
		warpAffine( img, dst, rot_mat, img.size() );
		
		// convert to gradient
		Mat gradient = getGradientSobel(dst);

		Mat hist = getQuadQuadHist(k,gradient);
		k.size /= 2;
		//
		hconcat(hist,getQuadQuadHist(k,gradient),hist);

		if(desc.cols == 0)
		{
			desc = hist.clone();
		}
		else
		{
			vconcat(desc,hist,desc);
		}
	}
	return desc;
}
*/
/*
Mat getHistofGradientManyPoints(Mat &img, vector<KeyPoint>& keypoints)
{
	// calculate histograms
	Mat gradient = getGradientSobel(img);

	Mat desc;
	for(int i = 0;i < keypoints.size();i++)
	{
		Mat hist = getQuadQuadHist(keypoints[i],gradient);
		keypoints[i].size /= 2;
		//
		hconcat(hist,getQuadQuadHist(keypoints[i],gradient),hist);

		if(desc.cols == 0)
		{
			desc = hist.clone();
		}
		else
		{
			vconcat(desc,hist,desc);
		}
	}
	return desc;
}
*/

#endif
