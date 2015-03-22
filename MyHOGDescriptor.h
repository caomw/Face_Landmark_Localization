
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


#ifndef MYHOG_H
#define MYHOG_H

#include <vector>
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

using namespace std;
using namespace cv;

// HOGDescriptor visual_imagealizer
// adapted for arbitrary size of feature sets and training images
Mat getHOGDescriptorVisualImage(Mat& origImg,
                                   vector<float>& descriptorValues,
                                   Size winSize,
                                   Size cellSize,                                   
                                   int scaleFactor,
                                   double viz_factor,
								   Point start = Point(0,0))
{   
	Mat visual_image;// = Mat::zeros(origImg.rows*scaleFactor,origImg.cols*scaleFactor, origImg.type()) + 255;
    resize(origImg, visual_image, Size(origImg.cols*scaleFactor, origImg.rows*scaleFactor));
	//imshow("test", visual_image);
	//cvWaitKey(0);
	//visual_image = visual_image * 0;
	//imshow("test", visual_image);
	//cvWaitKey(0);

    int gradientBinSize = 9;
    // dividing 180° into 9 bins, how large (in rad) is one bin?
    float radRangeForOneBin = 3.14/(float)gradientBinSize; 
 
    // prepare data structure: 9 orientation / gradient strenghts for each cell
	int cells_in_x_dir = winSize.width / cellSize.width;
    int cells_in_y_dir = winSize.height / cellSize.height;
    int totalnrofcells = cells_in_x_dir * cells_in_y_dir;
    float*** gradientStrengths = new float**[cells_in_y_dir];
    int** cellUpdateCounter   = new int*[cells_in_y_dir];
    for (int y=0; y<cells_in_y_dir; y++)
    {
        gradientStrengths[y] = new float*[cells_in_x_dir];
        cellUpdateCounter[y] = new int[cells_in_x_dir];
        for (int x=0; x<cells_in_x_dir; x++)
        {
            gradientStrengths[y][x] = new float[gradientBinSize];
            cellUpdateCounter[y][x] = 0;
 
            for (int bin=0; bin<gradientBinSize; bin++)
                gradientStrengths[y][x][bin] = 0.0;
        }
    }
 
    // nr of blocks = nr of cells - 1
    // since there is a new block on each cell (overlapping blocks!) but the last one
    int blocks_in_x_dir = cells_in_x_dir - 1;
    int blocks_in_y_dir = cells_in_y_dir - 1;
 
    // compute gradient strengths per cell
    int descriptorDataIdx = 0;
    int cellx = 0;
    int celly = 0;
 
    for (int blockx=0; blockx<blocks_in_x_dir; blockx++)
    {
        for (int blocky=0; blocky<blocks_in_y_dir; blocky++)            
        {
            // 4 cells per block ...
            for (int cellNr=0; cellNr<4; cellNr++)
            {
                // compute corresponding cell nr
                int cellx = blockx;
                int celly = blocky;
                if (cellNr==1) celly++;
                if (cellNr==2) cellx++;
                if (cellNr==3)
                {
                    cellx++;
                    celly++;
                }
 
                for (int bin=0; bin<gradientBinSize; bin++)
                {
                    float gradientStrength = descriptorValues[ descriptorDataIdx ];
                    descriptorDataIdx++;
 
                    gradientStrengths[celly][cellx][bin] += gradientStrength;
 
                } // for (all bins)
 
 
                // note: overlapping blocks lead to multiple updates of this sum!
                // we therefore keep track how often a cell was updated,
                // to compute average gradient strengths
                cellUpdateCounter[celly][cellx]++;
 
            } // for (all cells)
 
 
        } // for (all block x pos)
    } // for (all block y pos)
 
 
    // compute average gradient strengths
    for (int celly=0; celly<cells_in_y_dir; celly++)
    {
        for (int cellx=0; cellx<cells_in_x_dir; cellx++)
        {
 
            float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];
 
            // compute average gradient strenghts for each gradient bin direction
            for (int bin=0; bin<gradientBinSize; bin++)
            {
                gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
            }
        }
    }
 
 
    cout << "descriptorDataIdx = " << descriptorDataIdx << endl;
 
    // draw cells
    for (int celly=0; celly<cells_in_y_dir; celly++)
    {
        for (int cellx=0; cellx<cells_in_x_dir; cellx++)
        {
            int drawX = cellx * cellSize.width;
            int drawY = celly * cellSize.height;
 
            int mx = drawX + cellSize.width/2;
            int my = drawY + cellSize.height/2;
 
            rectangle(visual_image,
                      Point(drawX*scaleFactor,drawY*scaleFactor) + start,
                      Point((drawX+cellSize.width)*scaleFactor,
                      (drawY+cellSize.height)*scaleFactor) + start,
                      CV_RGB(255,255,255),
                      1);
 
            // draw in each cell all 9 gradient strengths
            for (int bin=0; bin<gradientBinSize; bin++)
            {
                float currentGradStrength = gradientStrengths[celly][cellx][bin];
 
                // no line to draw?
                if (currentGradStrength==0)
                    continue;
 
                float currRad = bin * radRangeForOneBin + radRangeForOneBin/2;
 
                float dirVecX = cos( currRad );
                float dirVecY = sin( currRad );
                float maxVecLen = cellSize.width/2;
                float scale = viz_factor; // just a visual_imagealization scale,
                                          // to see the lines better
 
                // compute line coordinates
                float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
                float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
                float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
                float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;
 
                // draw gradient visual_imagealization
                line(visual_image,
                     Point(x1*scaleFactor,y1*scaleFactor) + start,
                     Point(x2*scaleFactor,y2*scaleFactor) + start,
                     CV_RGB(255,255,255),
                     1);
 
            } // for (all bins)
 
        } // for (cellx)
    } // for (celly)
 
 
    // don't forget to free memory allocated by helper data structures!
    for (int y=0; y<cells_in_y_dir; y++)
    {
      for (int x=0; x<cells_in_x_dir; x++)
      {
           delete[] gradientStrengths[y][x];            
      }
      delete[] gradientStrengths[y];
      delete[] cellUpdateCounter[y];
    }
    delete[] gradientStrengths;
    delete[] cellUpdateCounter;
 
    return visual_image;
 
}

// computing HOG descriptor
void computeHOGDescriptor(Mat img_raw)
{
	resize(img_raw, img_raw, Size(img_raw.cols,img_raw.rows) );
 
	Mat img;
	cvtColor(img_raw, img, CV_RGB2GRAY);
 
 
	HOGDescriptor d;
	d.winSize = Size(16, 16);
	d.cellSize = Size(8, 8);
	// Size(128,64), //winSize
	// Size(16,16), //blocksize
	// Size(8,8), //blockStride,
	// Size(8,8), //cellSize,
	// 9, //nbins,
	// 0, //derivAper,
	// -1, //winSigma,
	// 0, //histogramNormType,
	// 0.2, //L2HysThresh,
	// 0 //gammal correction,
	// //nlevels=64
	//);
 
	// void HOGDescriptor::compute(const Mat& img, vector<float>& descriptors,
	//                             Size winStride, Size padding,
	//                             const vector<Point>& locations) const
	vector<float> descriptorsValues;
	vector<Point> locations;
	Point p = Point(10,10);
	Point p2 = Point(100,100);
	locations.push_back(p);
	d.compute( img, descriptorsValues, Size(0,0), Size(0,0), locations);
 
	cout << "HOG descriptor size is " << d.getDescriptorSize() << endl;
	cout << "img dimensions: " << img.cols << " width x " << img.rows << "height" << endl;
	cout << "Found " << descriptorsValues.size() << " descriptor values" << endl;
	cout << "Nr of locations specified : " << locations.size() << endl;

	Mat tmp = getHOGDescriptorVisualImage(img_raw,descriptorsValues,d.winSize,d.cellSize,1,1,p);
	//resize(tmp, tmp, Size(tmp.cols * 4,tmp.rows * 4) );
	imshow("tmp", tmp);
	cvWaitKey(0);
}

 Mat computeHOGDescriptor(Mat &img, vector<KeyPoint>& keypoints)
 {
	 Mat gray;
	 cvtColor(img, gray, CV_RGB2GRAY);
	 HOGDescriptor d;
	 d.winSize = Size(24, 24);
	 d.cellSize = Size(8, 8);
	 Mat desc = Mat::zeros(keypoints.size(), 36 * 6, CV_32F);
	 for(int i = 0;i < keypoints.size();i++)
	 {
		 KeyPoint k = keypoints[i];
		 Mat rot_mat = getRotationMatrix2D(k.pt,k.angle,1);
		 /// Rotate the gray image
	 	 Mat dst;
		 warpAffine( gray, dst, rot_mat, gray.size() );

		 vector<float> descriptorsValues;
		 vector<Point> locations;
		 locations.push_back(k.pt - Point2f(d.winSize.width / 2,d.winSize.height / 2));
		 d.compute( dst, descriptorsValues, Size(0,0), Size(0,0), locations);
		 memcpy(((float*)desc.data) + (i * desc.cols), &descriptorsValues[0], descriptorsValues.size() * sizeof(float));
		// visualize
		//Mat tmp = getHOGDescriptorVisualImage(dst,descriptorsValues,d.winSize,d.cellSize,1,1,locations[0]);
		//imshow("tmp", tmp);
		//cvWaitKey(0);
	 }

	 return desc;
 }

Mat computeHOGDescriptor(Mat &img, double minx, double miny,double maxx, double maxy,int& w,int& h)
{
	Mat gray;
	cvtColor(img, gray, CV_RGB2GRAY);
	HOGDescriptor d;
	w = ((int)(maxx - minx + 7) / 8) * 8;
	w = (w == 8) ? 16 : w;
	h = ((int)(maxy - miny + 7) / 8) * 8;
	h = (h == 8) ? 16 : h;
	d.winSize = Size(w, h);
	d.cellSize = Size(8, 8);
	Mat desc = Mat::zeros(1, 36 * (w / 8 - 1) * (h / 8 - 1), CV_32F);
	 
	vector<float> descriptorsValues;
	vector<Point> locations;
	locations.push_back(Point(minx,miny));
	d.compute( gray, descriptorsValues, Size(0,0), Size(0,0), locations);
	memcpy(((float*)desc.data), &descriptorsValues[0], descriptorsValues.size() * sizeof(float));
	 
	// visualize
	Mat tmp = getHOGDescriptorVisualImage(img,descriptorsValues,d.winSize,d.cellSize,1,1,locations[0]);
	imshow("tmp", tmp);
	cvWaitKey(0);

	return desc;
}

#endif
