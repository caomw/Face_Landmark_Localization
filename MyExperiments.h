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


#include "MyTraining.h"

void printMat(Mat m)
{
	for(int r = 0;r < m.rows;r++)
	{
		for(int c = 0;c < m.cols;c++)
		{
			cout << m.at<float>(cv::Point(c, r)) << "\t";
		}
		cout << endl;
	}
}

void testLaplace()
{
	Mat src, src_gray, dst;
  int kernel_size = 3;
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;
  char* window_name = "Laplace Demo";

  int c;

  /// Load an image
  src = imread( "D:/img.jpg" );

  if( !src.data )
    { return; }

  /// Remove noise by blurring with a Gaussian filter
  GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

  /// Convert the image to grayscale
  cvtColor( src, src_gray, CV_RGB2GRAY );

  /// Create window
  namedWindow( window_name, CV_WINDOW_AUTOSIZE );

  /// Apply Laplace function
  Mat abs_dst;

  Laplacian( src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( dst, abs_dst );

  /// Show what you got
  imshow( window_name, abs_dst );

  waitKey(0);

}

void testOpticalFlow()
{
	VideoCapture cap(0); // open the default camera
	if(!cap.isOpened()) // check if we succeeded
	{
		cout << "Error in Camera\n";
		return;
	}
	Mat next;
	string window_name = "video";
	namedWindow(window_name.c_str(),1);
	Mat prev;
	std::vector<Point2f> features_prev, features;
	for(int i = 0;;i++)
	{
		Mat frame;

		cap >> frame; // get a new frame from camera

		//next = getGradientSobel(frame);
		cvtColor( frame, next, CV_RGB2GRAY );
		

		if(i)
		{
			cv::resize(prev, prev, cv::Size(prev.cols + 10, prev.rows + 10));
			// Find position of feature in new image
			//vector<char> featuresFound;
			Mat featuresFound, err(next.rows, next.cols,CV_8UC3);
			cv::calcOpticalFlowPyrLK(
			  prev, next, // 2 consecutive images
			  features_prev, // input point positions in first im
			  features, // output point positions in the 2nd
			  featuresFound,    // tracking success
			  err,       // tracking error
			cv::Size(50,50)
			  );
			
			for(size_t i=0; i<features.size(); i++){
				if(featuresFound.at<char>(cv::Point(0,i))){
					line(frame,features[i],features_prev[i],Scalar(0,0,255));
                }
            }
			imshow(window_name.c_str(), frame);
		}
		else
		{			
			// Obtain initial set of features
			cv::goodFeaturesToTrack(next, // the image 
			  features,   // the output detected features
			  100,  // the maximum number of features 
			  0.01,     // quality level
			  10     // min distance between two features
			);
		}
		if(waitKey(30) >= 0) break;
		prev = next;
		features_prev = features;
		features.clear();
	}
}


void testVideoGradient()
{
	VideoCapture cap(0); // open the default camera
	if(!cap.isOpened()) // check if we succeeded
	{
		cout << "Error in Camera\n";
		return;
	}
	Mat next;
	string window_name = "video";
	namedWindow(window_name.c_str(),1);
	Mat prev;
	std::vector<Point2f> features_prev, features;
	for(int i = 0;;i++)
	{
		Mat frame;

		cap >> frame; // get a new frame from camera

		frame = getGradientSobel(frame);
		imshow(window_name.c_str(),frame);
		cvWaitKey(50);
	}
}
void trainOpticalFlowFidPoint(string Folder, string type)
{
	ostringstream str;
	// reading the file describing all images
	str << Folder << type << "\\" << type << ".txt";
	string file = str.str();
	ifstream inp(file.c_str());
	string line;
	// find the correct files
	//ofstream correct(Folder + type + "\\correct.txt");
	// read correct files only
	string tmpStr = Folder + type + "\\correct.txt";
	ifstream correct(tmpStr.c_str());

	printf("Loading...");
	std::vector<Point2f> features_prev, features;
	Mat prev, next;

	while(getline(inp,line,'\n'))
	{
		int readFlag;
		correct >> readFlag;
		if(!readFlag)
			continue;

		istringstream lineInp(line);
		vector<string> v; // all images
		string tmp;
		while(lineInp >> tmp)
		{
			v.push_back(tmp);
		}
		// display images only now
		string imgName = "";
		double minx = 1000000, miny = 1000000, maxx = 0,maxy = 0;
		try
		{
			imgName = Folder + type + "\\" + v[0] + "\\img." + v[2];
			//cout << "Loading img " << imgName << endl;
			printf(".");
			cv::Mat img = cv::imread(imgName);
			// load the labels
			string labels = Folder + type + "\\" + v[0] + "\\average.txt";
			// load fiducial points from file
			vector<Fiducial> fids = loadFiducial(labels, minx, miny, maxx, maxy);

			// scale to inter_ocular = 55
			// find the face rectangle
			double dx,dy;
			getFaceRectangle(img,minx,miny,maxx,maxy,dx,dy);
			// get the face rectangle
			img = img(cvRect(minx,miny,dx,dy));
			double interOccular = abs(fids[16].x - fids[17].x);
			double ratio = 55.0 / interOccular;
			cv::resize(img,img,cvSize(dx*ratio,dy*ratio));
			// move the fid points as well to (minx, miny) and resize them with the same inter_ocular ratio
			for(int i = 0;i < fids.size();i++)
			{
				fids[i].x = (fids[i].x - minx) * ratio;
				fids[i].y = (fids[i].y - miny) * ratio;
			}


			/*
			next = getGradientSobel(img);
			// optical flow
			std::vector<KeyPoint> keypoints;
			for(int i = 0;i < fids.size();i++)
			{
				keypoints.push_back(getKeyPoint(fids[i].x,fids[i].y,7,0));
				keypoints[i].response = 4;
			}
			FREAK extractor(false);
			Mat desc;
			extractor.compute(next, keypoints, desc);
			for(int i = 0;i < keypoints.size();i++)
			{
				cv::circle(img, keypoints[i].pt, 2, cv::Scalar(255,255,255), -1);
				cv::circle(img, keypoints[i].pt, 3, cv::Scalar(0,0,0), 1);
			}*/
			// convert to gray
			Mat gray;
			cvtColor( img, gray, CV_RGB2GRAY );
			Mat dst;
			// get bp
			lbp::OLBP(gray, dst);
			normalize(dst, dst, 0, 255, NORM_MINMAX, CV_8UC1);
			//imshow("eye",dst(cv::Rect(fids[0].x - 14, fids[0].y - 14,29,29)));
			//cvWaitKey(0);
			// calculate histogram
			Mat hist = lbp::histogram(dst(cv::Rect(fids[0].x - 14, fids[0].y - 14,28,28)), 256);
			Mat hist2 = Mat::ones(512, 512, CV_8UC1);
			hist2 = 255 * hist2;
			for(int i = 0;i < 256;i++)
			{
				cv::Point p0 = cv::Point(i*2, 0);
				cv::Point p = cv::Point(i*2, hist.at<int>(0,i));
				cout << hist.at<int>(0,i) << "\t";
				cv::line(hist2, p0, p,cv::Scalar(0,0,255));
			}
			cout << endl;
			//normalize(dst, dst, 0, 1, NORM_MINMAX, CV_8UC1);
			//dst = 255 * dst;
			//dst = getGradientSobel(dst);
			imshow("test", hist2);
			cvWaitKey(0);
			//cout << desc.rows << endl;
		}
		catch(...)
		{
			//correct << "0\n";
			cout << imgName << endl;
			cout << "This image is corrupted or doesn't exist\n";
		}
	}
	correct.close();
	puts("Done!");
}

Mat calculateSuperPixel(Mat img)
{
	image<rgb> *input = new image<rgb>(img.cols,img.rows,false);
	// copy image to the new structure to process
	vector<cv::Mat> channels;
	cv::split(img, channels);
	for (int y = 0; y < img.rows; y++) 
	{
		for (int x = 0; x < img.cols; x++) 
		{

			imRef(input,x,y).r = channels[2].at<unsigned char>(cv::Point(x,y));
			imRef(input,x,y).g = channels[1].at<unsigned char>(cv::Point(x,y));
			imRef(input,x,y).b = channels[0].at<unsigned char>(cv::Point(x,y));
		}
	}
	//
	int num_ccs; 
	image<rgb> *seg = segment_image(input, 0.2, 1000, 20, &num_ccs); 

	// copy the imageback
	for (int y = 0; y < img.rows; y++) 
	{
		for (int x = 0; x < img.cols; x++) 
		{

			channels[2].at<unsigned char>(cv::Point(x,y)) = imRef(seg,x,y).r;
			channels[1].at<unsigned char>(cv::Point(x,y)) = imRef(seg,x,y).g;
			channels[0].at<unsigned char>(cv::Point(x,y)) = imRef(seg,x,y).b;
		}
	}
	// merge the channels
	cv::merge(channels, img);
	// return the segmented image
	return img;
}
