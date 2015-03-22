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


#ifndef MYGA_H
#define MYGA_H

#include "MyTesting.h"
#include "MySVMProcessor.h"
#include <cmath>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
//#include <direct.h>

#define UNDEFINED 999999
#define TEST_PART 0

struct Candidate
{
	vector<DetecetedPoint> features;
	double fit;
};

// divisions to use with sorted data
struct Division
{
	int start; // starting index in the fids
	int size; // size of this division
	Candidate best_candidate;
};

class GA
{
	vector<vector<double> > response[29];
	DetecetedPoint bestPoints[29];
	vector<vector<vector<float> > > desc;
	Mat img1;
	Mat gray1;
	int imgWidth;
	int imgHeight;
	double sizeFactor;
	multimap<double,Candidate> pop; // population
	vector<vector<Fiducial> > all_fids; // all fiducial points (random till now)
	vector<double> avg_dists; // mean normalized distances between face parts
	vector<double> var_dists; // variance for normalized distances for face parts
	vector<MySVMS*> svm_poses; // for all fids svms
	MySVMProcessor *allPartsAngles; // SVM for all parts angles
	vector<vector<int> > angles; // angle points for all parts
	vector<DetecetedPoint> bestResponse;
	vector<int> bestFidNum;
	double bestRotationAngle;
	double maxx;
	vector<Division> div_fids;
	bool calcDivIndex;
	int div_ind;

	static GA*inst;
	GA(){sizeFactor = 1.0/8.0;};
public:
	static GA* getInst()
	{
		if(inst == 0)
		{
			inst = new GA;
		}
		return inst;
	}

	// make sure the response at the specified point is calculated
	void calcResponse(int fidInd, double x, double y)
	{
		// if point outside boundaries 
		// move the point to the nearest border and get the value there
		if(x < 0)
		{
			x = 0;
		}
		else if(x >= imgWidth)
		{
			x = imgWidth - 1;
		}
		if(y < 0)
		{
			y = 0;
		}
		else if(y>= imgHeight)
		{
			y = imgHeight - 1;
		}

		// if the respone is undifiend
		if(GA::response[fidInd][y][x] == UNDEFINED)
		{
			// if this point is not present in the descriptors
			if(!desc[y][x].size())
			{
				desc[y][x] = MySIFT::getSiftDescriptor(gray1, Testing::getKeyPoint(x/sizeFactor,y/sizeFactor,0));
			}
			double best_reponse = -1000000;
			for(int i = 0;i < svm_poses.size();i++)
			{
				best_reponse = max(best_reponse,svm_poses[i]->calcResponse(desc[y][x],fidInd));
			}
			GA::response[fidInd][y][x] = best_reponse;
			//maxx = max(maxx, GA::response[fidInd][y][x]);
			if(best_reponse > bestPoints[fidInd].prob)
			{
				bestPoints[fidInd].prob = best_reponse;
				bestPoints[fidInd].x = x;
				bestPoints[fidInd].y = y;
			}
		}
	}

	// get response at one point for one fiducial point type
	double getResponse(int ind,int y,int x)
	{
		if(ind < 0 || ind >= 29)
		{
			return 0;
		}
		// move the point to the nearest border and get the value there
		if(x < 0)
		{
			x = 0;
		}
		else if(x >= imgWidth)
		{
			x = imgWidth - 1;
		}
		if(y < 0)
		{
			y = 0;
		}
		else if(y>= imgHeight)
		{
			y = imgHeight - 1;
		}
		return this->response[ind][y][x];
	}

	// rotate a candidate in 2d with random angle +/- degree
	void affine_transform_candidate_random(Candidate &c1,int degree,int percent,int pixels)
	{
		vector<DetecetedPoint> tmp(c1.features.size());
		// get the center
		double x_c = 0;
		double y_c = 0;
		for(int i = 0;i < tmp.size();i++)
		{
			x_c += c1.features[i].x;
			y_c += c1.features[i].y;
		}
		x_c /= tmp.size();
		y_c /= tmp.size();

		bool flag = false;
		do
		{
			flag = false;
			double theta = (rand() % (degree*2 + 1)) - degree;
			double cos_theta = cos(theta * PI / 180);
			double sin_theta = sin(theta * PI / 180);
			double factor = (rand() % (percent*2 + 1)) - percent + 100;
			factor /= 100.0;
			int dx = (rand() % (2*pixels + 1)) - pixels;
			int dy = (rand() % (2*pixels + 1)) - pixels;
			for(int i = 0;i < tmp.size() && !flag;i++)
			{
				// translate the point then rotate then translate back
				tmp[i].x = factor * ((c1.features[i].x - x_c) * cos_theta - (c1.features[i].y - y_c) * sin_theta) + x_c + dx;
				tmp[i].y = factor * ((c1.features[i].x - x_c) * sin_theta + (c1.features[i].y - y_c) * cos_theta) + y_c + dy;
				// if x or y out of boundary then break this loop
				/*if(tmp[i].x < 0 || tmp[i].x >= imgWidth)
				{
					flag = true;
				}
				else if(tmp[i].y < 0|| tmp[i].y >= imgHeight)
				{
					flag = true;
				}*/
			}
		}while(flag);
		// then copy the vector back to the features
		c1.features = tmp;
		for(int i = 0;i < tmp.size();i++)
		{
			int y = tmp[i].y;
			int x = tmp[i].x;
			calcResponse(i,x,y);
			c1.features[i].prob = this->getResponse(i,y,x);
		}
	}

	// rotate a candidate in 2d with random angle +/- degree
	void affine_transform_part_random(Candidate &c2,vector<int> pts,int degree,int percent,int pixels)
	{
		vector<DetecetedPoint> tmp(pts.size());
		// get the center
		double x_c = 0;
		double y_c = 0;
		for(int i = 0;i < tmp.size();i++)
		{
			x_c += c2.features[pts[i]].x;
			y_c += c2.features[pts[i]].y;
		}
		x_c /= tmp.size();
		y_c /= tmp.size();

		bool flag = false;
		do
		{
			flag = false;
			double theta = (rand() % (degree*2 + 1)) - degree;
			double cos_theta = cos(theta * PI / 180);
			double sin_theta = sin(theta * PI / 180);
			double factor = (rand() % (percent*2 + 1)) - percent + 100;
			factor /= 100.0;
			int dx = (rand() % (2*pixels + 1)) - pixels;
			int dy = (rand() % (2*pixels + 1)) - pixels;
			for(int i = 0;i < tmp.size() && !flag;i++)
			{
				// translate the point then rotate then translate back
				tmp[i].x = factor * ((c2.features[pts[i]].x - x_c) * cos_theta - (c2.features[pts[i]].y - y_c) * sin_theta) + x_c + dx;
				tmp[i].y = factor * ((c2.features[pts[i]].x - x_c) * sin_theta + (c2.features[pts[i]].y - y_c) * cos_theta) + y_c + dy;
				// if x or y out of boundary then break this loop
				/*if(tmp[i].x < 0 || tmp[i].x >= imgWidth)
				{
					flag = true;
				}
				else if(tmp[i].y < 0|| tmp[i].y >= imgHeight)
				{
					flag = true;
				}*/
			}
		}while(flag);
		// then copy the vector back to the features
		for(int i = 0;i < tmp.size();i++)
		{
			int y = c2.features[pts[i]].y = tmp[i].y;
			int x = c2.features[pts[i]].x = tmp[i].x;
			calcResponse(pts[i],x,y);
			c2.features[pts[i]].prob = this->getResponse(pts[i],y,x);
		}
	}

	// rotate a candidate in 2d with random angle +/- degree
	void rotate_candidate_random(Candidate &c1,int degree)
	{
		vector<DetecetedPoint> tmp(c1.features.size());
		// get the center
		double x_c = 0;
		double y_c = 0;
		for(int i = 0;i < tmp.size();i++)
		{
			x_c += c1.features[i].x;
			y_c += c1.features[i].y;
		}
		x_c /= tmp.size();
		y_c /= tmp.size();

		bool flag = false;
		do
		{
			flag = false;
			double theta = (rand() % (degree*2 + 1)) - degree;
			double cos_theta = cos(theta * PI / 180);
			double sin_theta = sin(theta * PI / 180);
			for(int i = 0;i < tmp.size() && !flag;i++)
			{
				// translate the point then rotate then translate back
				tmp[i].x = ((c1.features[i].x - x_c) * cos_theta - (c1.features[i].y - y_c) * sin_theta) + x_c;
				tmp[i].y = ((c1.features[i].x - x_c) * sin_theta + (c1.features[i].y - y_c) * cos_theta) + y_c;
				// if x or y out of boundary then break this loop
				/*if(tmp[i].x < 0 || tmp[i].x >= imgWidth)
				{
					flag = true;
				}
				else if(tmp[i].y < 0|| tmp[i].y >= imgHeight)
				{
					flag = true;
				}*/
			}
		}while(flag);
		// then copy the vector back to the features
		c1.features = tmp;
		for(int i = 0;i < tmp.size();i++)
		{
			int y = tmp[i].y;
			int x = tmp[i].x;
			calcResponse(i,x,y);
			c1.features[i].prob = this->getResponse(i,y,x);
		}
	}

	void rotate_part_random(Candidate &c2,vector<int>& pts,int degree)
	{
		vector<DetecetedPoint> tmp(pts.size());
		// get the center
		double x_c = 0;
		double y_c = 0;
		for(int i = 0;i < tmp.size();i++)
		{
			x_c += c2.features[pts[i]].x;
			y_c += c2.features[pts[i]].y;
		}
		x_c /= tmp.size();
		y_c /= tmp.size();

		bool flag = false;
		do
		{
			flag = false;
			double theta = (rand() % (degree*2 + 1)) - degree;
			double cos_theta = cos(theta * PI / 180);
			double sin_theta = sin(theta * PI / 180);
			for(int i = 0;i < tmp.size() && !flag;i++)
			{
				// translate the point then rotate then translate back
				tmp[i].x = ((c2.features[pts[i]].x - x_c) * cos_theta - (c2.features[pts[i]].y - y_c) * sin_theta) + x_c;
				tmp[i].y = ((c2.features[pts[i]].x - x_c) * sin_theta + (c2.features[pts[i]].y - y_c) * cos_theta) + y_c;
				// if x or y out of boundary then break this loop
				/*if(tmp[i].x < 0 || tmp[i].x >= imgWidth)
				{
					flag = true;
				}
				else if(tmp[i].y < 0|| tmp[i].y >= imgHeight)
				{
					flag = true;
				}*/
			}
		}while(flag);
		// then copy the vector back to the features
		for(int i = 0;i < tmp.size();i++)
		{
			int y = c2.features[pts[i]].y = tmp[i].y;
			int x = c2.features[pts[i]].x = tmp[i].x;
			calcResponse(pts[i],x,y);
			c2.features[pts[i]].prob = this->getResponse(pts[i],y,x);
		}
	}

	// scale a candidate in 2d with random percentage +/- percent
	void scale_candidate_random(Candidate &c1,int percent)
	{
		vector<DetecetedPoint> tmp(c1.features.size());
		// get the center
		double x_c = 0;
		double y_c = 0;
		for(int i = 0;i < tmp.size();i++)
		{
			x_c += c1.features[i].x;
			y_c += c1.features[i].y;
		}
		x_c /= tmp.size();
		y_c /= tmp.size();

		bool flag = false;
		do
		{
			flag = false;
			double factor = (rand() % (percent*2 + 1)) - percent + 100;
			factor /= 100.0;
			for(int i = 0;i < tmp.size() && !flag;i++)
			{
				// translate the point then rotate then translate back
				tmp[i].x = ((c1.features[i].x - x_c) * factor) + x_c;
				tmp[i].y = ((c1.features[i].y - y_c) * factor) + y_c;
				// if x or y out of boundary then break this loop
				/*if(tmp[i].x < 0 || tmp[i].x >= imgWidth)
				{
					flag = true;
				}
				else if(tmp[i].y < 0|| tmp[i].y >= imgHeight)
				{
					flag = true;
				}*/
			}
		}while(flag);
		// then copy the vector back to the features
		c1.features = tmp;
		for(int i = 0;i < tmp.size();i++)
		{
			int y = tmp[i].y;
			int x = tmp[i].x;
			calcResponse(i,x,y);
			c1.features[i].prob = this->getResponse(i,y,x);
		}
	}

	// scale a part in 2d with random percentage +/- percent
	void scale_part_random(Candidate &c2,vector<int> & pts,int percent)
	{
		vector<DetecetedPoint> tmp(pts.size());
		// get the center
		double x_c = 0;
		double y_c = 0;
		for(int i = 0;i < tmp.size();i++)
		{
			x_c += c2.features[pts[i]].x;
			y_c += c2.features[pts[i]].y;
		}
		x_c /= tmp.size();
		y_c /= tmp.size();

		bool flag = false;
		do
		{
			flag = false;
			double factor = (rand() % (percent*2 + 1)) - percent + 100;
			factor /= 100.0;
			for(int i = 0;i < tmp.size() && !flag;i++)
			{
				// translate the point then rotate then translate back
				tmp[i].x = ((c2.features[pts[i]].x - x_c) * factor) + x_c;
				tmp[i].y = ((c2.features[pts[i]].y - y_c) * factor) + y_c;
				// if x or y out of boundary then break this loop
				/*if(tmp[i].x < 0 || tmp[i].x >= imgWidth)
				{
					flag = true;
				}
				else if(tmp[i].y < 0|| tmp[i].y >= imgHeight)
				{
					flag = true;
				}*/
			}
		}while(flag);
		// then copy the vector back to the features
		for(int i = 0;i < tmp.size();i++)
		{
			int y = c2.features[pts[i]].y = tmp[i].y;
			int x = c2.features[pts[i]].x = tmp[i].x;
			calcResponse(pts[i],x,y);
			c2.features[pts[i]].prob = this->getResponse(pts[i],y,x);
		}
	}

	// TRANSLATE a candidate in 2d with random distance +/- pixels (x or y)
	void translate_candidate_random(Candidate &c1,int pixels)
	{
		vector<DetecetedPoint> tmp(c1.features.size());

		bool flag = false;
		do
		{
			flag = false;
			int dx = (rand() % (2*pixels + 1)) - pixels;
			int dy = (rand() % (2*pixels + 1)) - pixels;
			for(int i = 0;i < tmp.size() && !flag;i++)
			{
				// translate the point then rotate then translate back
				tmp[i].x = c1.features[i].x + dx;
				tmp[i].y = c1.features[i].y + dy;
				// if x or y out of boundary then break this loop
				/*if(tmp[i].x < 0 || tmp[i].x >= imgWidth)
				{
					flag = true;
				}
				else if(tmp[i].y < 0|| tmp[i].y >= imgHeight)
				{
					flag = true;
				}*/
			}
		}while(flag);
		// then copy the vector back to the features
		c1.features = tmp;
		for(int i = 0;i < tmp.size();i++)
		{
			int y = tmp[i].y;
			int x = tmp[i].x;
			calcResponse(i,x,y);
			c1.features[i].prob = this->getResponse(i,y,x);
		}
	}

	// TRANSLATE a part in 2d with random distance +/- pixels (x or y)
	void translate_part_random(Candidate &c2,vector<int> pts,int pixels)
	{
		vector<DetecetedPoint> tmp(pts.size());

		bool flag = false;
		do
		{
			flag = false;
			int dx = (rand() % (2*pixels + 1)) - pixels;
			int dy = (rand() % (2*pixels + 1)) - pixels;
			for(int i = 0;i < tmp.size() && !flag;i++)
			{
				// translate the point then rotate then translate back
				tmp[i].x = c2.features[pts[i]].x + dx;
				tmp[i].y = c2.features[pts[i]].y + dy;
				// if x or y out of boundary then break this loop
				/*if(tmp[i].x < 0 || tmp[i].x >= imgWidth)
				{
					flag = true;
				}
				else if(tmp[i].y < 0|| tmp[i].y >= imgHeight)
				{
					flag = true;
				}*/
			}
		}while(flag);
		// then copy the vector back to the features
		for(int i = 0;i < tmp.size();i++)
		{
			//printf("(%llf,%llf) => (%llf,%llf)\n",c2.features[pts[i]].x,c2.features[pts[i]].y,tmp[i].x,tmp[i].y);
			int y = c2.features[pts[i]].y = tmp[i].y;
			int x = c2.features[pts[i]].x = tmp[i].x;
			calcResponse(pts[i],x,y);
			c2.features[pts[i]].prob = this->getResponse(pts[i],y,x);
		}
	}

	// get one random point with high response
	void getRandHighReponsePoint()
	{
		bestResponse.resize(1);
		bestFidNum.resize(1);
		bestResponse[0].prob = -10;
		bestFidNum[0] = -1;
		// starting time
		time_t first,last;
		time(&first);
		double val = 0.8;
		while(bestResponse[0].prob < val)
		{
			// get random point
			int x = rand() % imgWidth;
			int y = rand() % imgHeight;
			// get the response from all SVMs
			// loop on all points except the worst 3 points
			for(int i = 0;i < 29;i++)
			{
				calcResponse(i,x,y);
				if(GA::response[i][y][x] > bestResponse[0].prob)
				{
					maxx = max(maxx, GA::response[i][y][x]);
					bestResponse[0].prob = GA::response[i][y][x];
					bestResponse[0].x = x;
					bestResponse[0].y = y;
					bestFidNum[0] = i;
				}
			}
			time(&last);
			double seconds = difftime(last,first);
			if(seconds > 1) // more than one second
			{
				// restart timer
				time(&first);
				// relax the rule
				val -= 0.9;
			}
		}
	}

	// get one random point with high response
	void getRandHighReponsePoint2()
	{
		bestResponse.resize(2);
		bestFidNum.resize(2);
		bestResponse[1].prob = -10;
		bestFidNum[1] = -1;
		// starting time
		time_t first,last;
		time(&first);
		double val = 0.8;
		while(bestResponse[1].prob < val || bestFidNum[1] == bestFidNum[0])
		{
			// get random point
			int x = rand() % imgWidth;
			int y = rand() % imgHeight;
			// get the response from all SVMs
			// loop on all points except the worst 3 points
			for(int i = 0;i < 29;i++)
			{
				calcResponse(i,x,y);
				if(GA::response[i][y][x] > 0.8)
				{
					maxx = max(maxx, GA::response[i][y][x]);
					bestResponse[1].prob = GA::response[i][y][x];
					bestResponse[1].x = x;
					bestResponse[1].y = y;
					bestFidNum[1] = i;
				}
			}
			time(&last);
			double seconds = difftime(last,first);
			if(seconds > 1) // more than one second
			{
				// restart timer
				time(&first);
				// relax the rule
				val -= 0.9;
			}
		}
	}

	// intialize candidate from the orignal fids
	void set_candidate_from_fids(vector<Fiducial> fids,Candidate &c1)
	{
		// and minimum and maximum x and y
		Point2f pt_min(10000,10000);
		Point2f pt_max(0,0);
		for(int i = 0;i < fids.size();i++)
		{
			Point2f pt(fids[i].x,fids[i].y);
			pt_min.x = min(pt_min.x, pt.x);
			pt_min.y = min(pt_min.y, pt.y);
			pt_max.x = max(pt_max.x, pt.x);
			pt_max.y = max(pt_max.y, pt.y);
		}
		// create rectangle
		double width(pt_max.x - pt_min.x);
		double height(pt_max.y - pt_min.y);
		// then get the ratio needed
		double width_ratio = 1;
		double height_ratio = 1;
		//// taking only 80% from the image as the effective face image
		width_ratio = (imgWidth * 0.8) / width;
		height_ratio = (imgHeight * 0.8) / height;
		//// trying to take width and height ratios from two deteceted points
		//int k1 = rand() % bestFidNum.size();
		//int k2;
		//do
		//{
		//	k2 = rand() % bestFidNum.size();
		//}while(bestFidNum[k1] == bestFidNum[k2]);
		//// then take the ratio between these two points
		//// width
		//double dx_orig = abs(bestResponse[k1].x - bestResponse[k2].x) + 1;
		//double dx_target = abs(fids[k1].x - fids[k2].x) + 1;
		//width_ratio = dx_orig / dx_target;
		//// height
		//double dy_orig = abs(bestResponse[k1].y - bestResponse[k2].y) + 1;
		//double dy_target = abs(fids[k1].y - fids[k2].y) + 1;
		//height_ratio = dy_orig / dy_target;
		//width_ratio = height_ratio = min(width_ratio, height_ratio);
		Point2f pt_c(width*width_ratio*0.5,height*height_ratio*0.5);

		for(int i = 0;i < fids.size();i++)
		{
			fids[i].x = (fids[i].x - pt_min.x) * width_ratio;
			fids[i].y = (fids[i].y - pt_min.y) * height_ratio;
		}
		// image start at 10% from the width and height
		Point2f im_start(imgWidth,imgHeight);
		im_start.x /= 2;
		im_start.y /= 2;
		// then translate the points to the candidate with the center and the ratio
		for(int i = 0;i < fids.size();i++)
		{
			float x = fids[i].x - pt_c.x + im_start.x;
			//if(x < 0) x = 0; if(x >= imgWidth) x = imgWidth - 1;
			float y = fids[i].y - pt_c.y + im_start.y;
			//if(y < 0) y = 0; if(y >= imgHeight) y = imgHeight - 1;
			// then set the features
			DetecetedPoint dp;
			dp.x = x;
			dp.y = y;
			c1.features.push_back(dp);
		}
		// then randomly choose between center of the image and one of the best points
		
		if(rand()&1)
		{
			int dx, dy;
			// translate the featuers based on one of the points reponses
			int k = rand() % bestFidNum.size();
			// then translate the features according to this point
			dx = bestResponse[k].x - c1.features[bestFidNum[k]].x;
			dy = bestResponse[k].y - c1.features[bestFidNum[k]].y;
			for(int i = 0;i < fids.size();i++)
			{
				c1.features[i].x += dx;
				//if(c1.features[i].x < 0) c1.features[i].x = 0; 
				//else if(c1.features[i].x >= imgWidth) c1.features[i].x = imgWidth - 1;
				c1.features[i].y += dy;
				//if(c1.features[i].y < 0) c1.features[i].y = 0; 
				//else if(c1.features[i].y >= imgHeight) c1.features[i].y = imgHeight - 1;
			}
		}
		// then rotate them with the difference in angles based on the two best points
		affine_transform_candidate_random(c1,5,5,5);
		calc_fitness(c1);
		// then choose new best points
		for(int i = 0;i < c1.features.size();i++)
		{
			if(c1.features[i].prob > 0.8)
			{
				bestResponse.push_back(c1.features[i]);
				bestFidNum.push_back(i);
			}
		}
	}

	void insert_candidate(Candidate &c1)
	{
		pop.insert(make_pair(1000 - c1.fit, c1));
	}

	// calculate the division index based on the intial GA's and divisions
	void calc_div_ind()
	{
		vector<int> prob(div_fids.size());
		for(int i = 0;i < div_fids.size();i++)
		{
			double a = div_fids[i].best_candidate.fit;
			if(a < 1)
			{
				prob[i] = 1;
			}
			else
			{
				prob[i] = (a + 1);
			}
		}
		for(int i = 1;i < prob.size();i++)
		{
			prob[i] += prob[i - 1];
		}
		// then get prob
		int c = rand() % prob[prob.size() - 1];
		for(int i = 0;i < prob.size();i++)
		{
			if(c < prob[i])
			{
				div_ind = i;
				return;
			}
		}
	}

	// intialize the algorithm
	void init(Mat img,int population = 100,bool newImg = true,bool clearPopulation = true,
		bool calcHighResponse = true)
	{
		if(newImg)
		{
			sizeFactor = 1.0;
			maxx = 0;
			GA::img1 = img.clone();
			// convert to gray
			cvtColor( img, GA::gray1, CV_RGB2GRAY );
			GA::gray1.convertTo(GA::gray1,CV_32FC1); // convert to floating point to use in SIFT calculations
			// create the response maps
			imgHeight = img1.rows * sizeFactor;
			imgWidth = img1.cols * sizeFactor;
			vector<double> row;
			for(int i = 0;i < imgWidth;i++)
			{
				row.push_back(UNDEFINED);
			}
			vector<vector<double> > tmp;
			for(int i = 0;i < imgHeight;i++)
			{
				tmp.push_back(row);
			}
			for(int i = 0;i < 29;i++)
			{
				GA::response[i] = tmp;
			}
			// create the descriptors map
			vector<vector<float> > rowD(imgWidth);
			desc.clear();
			for(int i = 0;i < imgHeight;i++)
			{
				desc.push_back(rowD);
			}
			// init best points with prob = - 10
			for(int i = 0;i < 29;i++)
			{
				bestPoints[i].prob = -10;
			}
		}
		if(clearPopulation)
		{
			// clear population
			pop.clear();
		}
		// and create a new generation
		if(calcHighResponse)
		{
			// start by finding random points with high reponses for one of the SVMs
			puts("First Point");
			getRandHighReponsePoint();
			// get the second best point
			puts("Second Point");
			getRandHighReponsePoint2();
		}
		// create from the available fiducial data
		while(pop.size() < population)
		{
			// create new candidate
			Candidate c1;
			// set the feature points from old fids
			if(calcDivIndex)
			{
				calc_div_ind();
			}
			set_candidate_from_fids(all_fids[div_fids[div_ind].start + (rand()%div_fids[div_ind].size)],c1);
			// add the candidate to population
			insert_candidate(c1);
#if _VIS
		//ostringstream ostr;
		//ostr << "init_" << c1.fit << "_.jpg";
		//Mat tmpimg = Visualization::overlayFidsImage(this->img1,c1.features,3);
		//imwrite(ostr.str().c_str(),tmpimg);
#endif
		}
	}

	// re_intialize the algorithm by double size
	void re_init(int population)
	{
		// create the response maps
		for(int i = 0;i < 29;i++)
		{
			vector<vector<double> > tmp;
			for(int j = 0;j < imgHeight;j++)
			{
				vector<double> row0; // the old row
				vector<double> row1; // the new row
				for(int k = 0;k < imgWidth;k++)
				{
					row0.push_back(GA::response[i][j][k]);
					row0.push_back(UNDEFINED);
					//
					row1.push_back(UNDEFINED);
					row1.push_back(UNDEFINED);
				}
				tmp.push_back(row0);
				tmp.push_back(row1);
			}
			GA::response[i] = tmp;
		}
		// create the descriptors map
		vector<vector<vector<float> > > new_desc;
		vector<vector<float> > rowD1(imgWidth * 2); // the new row
		for(int i = 0;i < imgHeight;i++)
		{
			vector<vector<float> > rowD0(imgWidth * 2);
			for(int j = 0;j < imgWidth;j++)
			{
				rowD0[j * 2] = desc[i][j];
			}
			new_desc.push_back(rowD0);
			new_desc.push_back(rowD1);
		}
		desc = new_desc;
		// move the best points to the double distance
		// clear the best reponses
		bestResponse.clear();
		bestFidNum.clear();
		for(int i = 0;i < 29;i++)
		{
			bestPoints[i].x *= 2;
			bestPoints[i].y *= 2;
			//
			bestResponse.push_back(bestPoints[i]);
			bestFidNum.push_back(i);
		}
		// double the size
		sizeFactor *= 2;
		imgWidth *= 2;
		imgHeight *= 2;
		// double the candidate sizes
		for(multimap<double, Candidate>::iterator itr = pop.begin();itr != pop.end();itr++)
		{
			for(int i = 0;i < 29;i++)
			{
				itr->second.features[i].x *= 2;
				itr->second.features[i].y *= 2;
			}
		}
		// create the rest of the population from the available fiducial data
		while(pop.size() < population)
		{
			// create new candidate
			Candidate c1;
			// set the feature points from old fids
			set_candidate_from_fids(all_fids[div_fids[div_ind].start + (rand()%div_fids[div_ind].size)],c1);
			// add the candidate to population
			insert_candidate(c1);
		}
	}


	Candidate cross_over(Candidate c1,Candidate c2,
		vector<DetecetedPoint> &v1,vector<DetecetedPoint> &v2,vector<DetecetedPoint> &v3)
	{
		Candidate off_spring;
		off_spring.features = c1.features;
		int switchVal = rand()%5;
		
		switch(switchVal)
		{
		case 0:
			{
				// using general faceparts		
				int generalFaceParts[] = {7,17,21};
				int crossPoint = rand()%3;
				// before using the new part make sure it's response sum up to positive value
				/*double val1 = 0;
				for(int i = 0;i <= generalFaceParts[crossPoint];i++)
				{
					val1 += c1.features[i].prob;
				}*/
				double val2 = 0;
				for(int i = generalFaceParts[crossPoint] + 1;i < 29;i++)
				{
					val2 += c2.features[i].prob;
				}
				// if positive value then you can copy
				if(val2 > 0)
				{
					for(int i = generalFaceParts[crossPoint] + 1;i < 29;i++)
					{
						off_spring.features[i] = c2.features[i];
						v2.push_back(c2.features[i]);
					}
					for(int i = 0;i <= generalFaceParts[crossPoint];i++)
					{
						v1.push_back(c1.features[i]);
					}
				}
			}
			break;
		case 4:
			{
				// using general faceparts		
				int generalFaceParts[] = {8,18,22};
				int crossPoint = rand()%3;
				// before using the new part make sure it's response sum up to positive value
				double val2 = 0;
				for(int i = 0; i < generalFaceParts[crossPoint];i++)
				{
					val2 += c2.features[i].prob;
				}
				// if positive value then you can copy
				if(val2 > 0)
				{
					for(int i = 0; i < generalFaceParts[crossPoint];i++)
					{
						off_spring.features[i] = c2.features[i];
						v2.push_back(c2.features[i]);
					}
					for(int i = generalFaceParts[crossPoint];i < 29;i++)
					{
						v1.push_back(c1.features[i]);
					}
				}
			}
			break;
		case 1:
			{
				// using smaller face parts
				string parts[] = {"1 3 5 6 9 11 13 14 17",
								"2 4 7 8 10 12 15 16 18",
								"19 20 21 22",
								"23 24 25 26 27 28 29"};
				static vector<vector<int> > p = create_disjoint_parts(parts,4);
				vector<int> use_c2;
				for(int i = 0;i < p.size();i++)
				{
					use_c2.push_back(rand()&1);
				}
				double val2 = 0;
				for(int i = 0;i < p.size();i++)
				{
					if(use_c2[i])
					{
						val2 = 0;
						for(int k = 0;k < p[i].size();k++)
						{
							val2 += c2.features[p[i][k]].prob;
						}
						if(val2 > 0)
						{
							for(int k = 0;k < p[i].size();k++)
							{
								off_spring.features[p[i][k]] = c2.features[p[i][k]];
								v2.push_back(c2.features[p[i][k]]);
							}
						}
						else
						{
							for(int k = 0;k < p[i].size();k++)
							{
								v1.push_back(c1.features[p[i][k]]);
							}
						}
					}
					else
					{
						for(int k = 0;k < p[i].size();k++)
						{
							v1.push_back(c1.features[p[i][k]]);
						}
					}
				}
				
			}
			break;
		case 2:
			{
				// using smaller points
				string parts[] = {"1 2 9 10",
								 "3 4 11 12",
								 "5 6",
								 "7 8",
								 "13 14",
								"15 16",
								"17 18",
								"19 20",
								"21 22",
								"23 24 25 26",
								"23 24 27 28 29"};
				
				vector<vector<int> > p = create_disjoint_parts(parts,11);
				// randomly choose 5 differnt points X 8 times to make sure the points are connected
				// vector<vector<int> > p;
				for(int i = 0;i < p.size();i++)
				{
					set<int> s;
					for(int j = 0;j < p[i].size();j++)
					{
						s.insert(p[i][j]);
					}
					while(s.size() < 5)
					{
						s.insert(rand()%29);
					}
					vector<int> tmp;
					for(set<int>::iterator itr = s.begin();itr != s.end();itr++)
					{
						tmp.push_back(*itr);
					}
					//p.push_back(tmp);
					p[i] = tmp;
				}
				int m2[29];
				memset(m2,0,29 * sizeof(int));
				for(int i = 0;i < p.size();i++)
				{
					double f1= calc_fitness(c1,p[i]);
					double f2= calc_fitness(c2,p[i]);
					if(f2 > f1 && f2 > 0)
					{
						//cout << "Part." << i << endl;
						for(int k = 0;k < p[i].size();k++)
						{
							off_spring.features[p[i][k]] = c2.features[p[i][k]];
							m2[p[i][k]] = 1;
						}
						//cout << "End Part." << i << endl;
					}
				}
				for(int i = 0;i < 29;i++)
				{
					if(m2[i])
					{
						v2.push_back(c2.features[i]);
					}
					else
					{
						v1.push_back(c1.features[i]);
					}
				}
			}
			break;
		case 3:
			{
				// get ratio from each candidate (parts of 10)
				double c1_ratio = 0.5;//(rand()%9) + 1;
				double c2_ratio = 0.5;//10 - c1_ratio;
				//c1_ratio /= 10.0;
				//c2_ratio /= 10.0;
				// average
				for(int i = 0;i < c1.features.size();i++)
				{
					off_spring.features[i].x = (c1.features[i].x + c2.features[i].x) / 2;
					off_spring.features[i].y = (c1.features[i].y + c2.features[i].y) / 2;
					// v3.pu
					v3.push_back(off_spring.features[i]);
				}
				// calculate the new fitness
				//puts("before fitness loop");
				//static int cnt = 0;
				//cnt++;
				//cout << cnt << endl;
				for(int i = 0;i < c1.features.size();i++)
				{
					int y = off_spring.features[i].y;
					int x = off_spring.features[i].x;
					calcResponse(i,x,y);
					off_spring.features[i].prob = this->getResponse(i,y,x);
				}
				//puts("end of fitness calc");
			}
		}
		return off_spring;
	}

	// check for all available mutaion operations
	// (stretch some parts, 
	//  different rotations in-plane and out-of-plane, 
	//  moving parts up and down the main axis,
	// and may be hill climbing for some points)
	Candidate mutate(Candidate c1)
	{
		Candidate off_spring = c1;
		int w = 0;//rand()&1;
		int num = 10;
		if(w == 0)
		{
			int r = rand()%4; // choose between rotation, scaling, translation and full affine
			// change the whole face
			if(r == 0)
			{
				// rotate +/- 5 degrees random
				rotate_candidate_random(off_spring, num);
			}
			else if(r == 1)
			{
				// scale
				scale_candidate_random(off_spring, num);
			}
			else if(r == 2)
			{
				// translate
				translate_candidate_random(off_spring, num);
			}
			else
			{
				// full random affine
				affine_transform_candidate_random(off_spring,num,num,num);
			}
		}
		else if(w == 1)
		{
			// using smaller face parts
			string parts[] = {"1 3 5 6",
							"9 11 13 14 17",
							"2 4 7 8",
							"10 12 15 16 18",
							"19 20 21 22",
							"23 24 25 26 27 28 29"};
			int part = rand()%6;
			istringstream iss(parts[part]);
			int j;
			vector<int> pts;
			while(iss >> j)
			{
				pts.push_back(j - 1);
			}
			// random scaling and translation
			affine_transform_part_random(off_spring,pts,0,num, num);
		}
		return off_spring;
	}
	double getLength(double x1,double y1,double x2,double y2)
	{
		double dx = (x1 - x2) * (x1 - x2);
		double dy = (y1 - y2) * (y1 - y2);

		return sqrt(dx + dx);
	}
	
	vector<vector<int> > create_disjoint_parts(string parts[],int size)
	{
		vector<vector<int> > disjoint_parts;
		for(int i = 0;i < size;i++)
		{
			istringstream istr(parts[i]);
			vector<int> tmp;
			int j;
			while(istr >> j)
			{
				tmp.push_back(j - 1);
			}
			disjoint_parts.push_back(tmp);
		}
		return disjoint_parts;
	}
	
	// get the difference between to vertical lines across the face
	double get_angle_diff(vector<DetecetedPoint>& v)
	{
		int tmp_pts[] = {22,25,26,27,28,29};
		vector<int> pts(tmp_pts,tmp_pts + 6);

		int tmp_pts2[] = {3,4,19,20,23,24,29};
		vector<int> pts2(tmp_pts2,tmp_pts2 + 7);

		vector<float> line1,line2;
		vector<Point> line_pts;
		for(int j = 0;j < pts.size();j++)
		{
			line_pts.push_back(Point(v[pts[j] - 1].x,v[pts[j] - 1].y));
		}
		cv::fitLine(line_pts,line1,CV_DIST_L2,0,0.01,0.01); // parameter values from the documentations
		line_pts.clear();
		for(int j = 0;j < pts.size();j++)
		{
			line_pts.push_back(Point(v[pts[j] - 1].x,v[pts[j] - 1].y));
		}
		cv::fitLine(line_pts,line2,CV_DIST_L2,0,0.01,0.01); // parameter values from the documentations

		double angle1 = abs(atan2(line1[1], line1[0]) * 180 / PI);
		double angle2 = abs(atan2(line2[1], line2[0]) * 180 / PI);
		//cout << angle1 << "\t" << angle2 << endl;
		return abs(angle1 - angle2);
	}

	// check that the angle between horizontal fid points and vertical fid points is 90
	void check_angles_90(Candidate& c1)
	{
		if(c1.fit > 1)
		{
			double factor = 100;
			// then calculate the angles between points and update the fitness with ratio using the standard deviation of angles
			int parts[][2] = {{1,2},
			{3,4},
			{9,10},
			{11,12},
			{17,18},
			{19,20},
			{23,24}};
			vector<double> angles1(7);
			double avg1 = 0;
			for(int i = 0;i < angles1.size();i++)
			{
				angles1[i] = atan2(c1.features[parts[i][1] - 1].y - c1.features[parts[i][0] - 1].y,
					c1.features[parts[i][1] - 1].x - c1.features[parts[i][0] - 1].x);
				angles1[i]  = angles1[i] * 180 / PI;
				avg1 += angles1[i];
				//cout << angles[i] << endl;
			}
			avg1 /= angles1.size();
			// get std deviation
			double std_dev1 = 0;
			for(int i = 0;i < angles1.size();i++)
			{
				std_dev1 += ((angles1[i] - avg1) * (angles1[i] - avg1));
			}
			std_dev1 = sqrt(std_dev1 / angles1.size());
			// other opints
			int pts[] = {25,26,27,28,29};
			vector<double> angles2;
			double avg2 = 0;
			double std_dev2 = 0;
			for(int i = 0;i < 5;i++)
			{
				for(int j = i + 1;j < 5;j++)
				{
					angles2.push_back(atan2(c1.features[pts[j] - 1].y - c1.features[pts[i] - 1].y,
					c1.features[pts[j] - 1].x - c1.features[pts[i] - 1].x));
				}
			}
			for(int i = 0;i < angles2.size();i++)
			{
				angles2[i]  = angles2[i] * 180 / PI;
				avg2 += angles2[i];
				//cout << angles2[i] << endl;
			}
			avg2 /= angles2.size();
			for(int i = 0;i < angles2.size();i++)
			{
				std_dev2 += ((angles2[i] - avg2) * (angles2[i] - avg2));
			}
			std_dev2 = sqrt(std_dev2 / angles2.size());
			//std_dev2 = (std_dev2 < 10) ? 0 : std_dev2 - 10;
			factor = factor - std_dev1 - std_dev2;
			double diff = abs(abs(avg2 - avg1) - 90.0);
			if(diff > 5)
			{
				diff -= 5;
				factor -= diff;
			}
			if(factor < 1)
			{
				factor = 1;
			}
			c1.fit *= (factor / 100);
		}
	}

	void test_convex_hull(Candidate& c1)
	{
		string parts[] = {"1 3 5 6 9 11 13 14 17",
								"2 4 7 8 10 12 15 16 18",
								"19 20 21 22",
								"23 24 25 26 27 28",
								"29"};
		static vector<vector<int> > p = create_disjoint_parts(parts,5);
		// get convex hull for each part and make sure that no part inside the other parts
		int countInside = 0;
		for(int i = 0;i < p.size();i++)
		{
			vector<Point> pts; // the points
			vector<Point> ch; // the convex hull
			for(int k = 0;k < p[i].size();k++)
			{
				pts.push_back(Point(c1.features[p[i][k]].x,c1.features[p[i][k]].y));
			}
			convexHull(pts,ch);
			//cout << "Convex Hull " << i << " Size = " << ch.size() << endl;
			for(int j = 0;j < p.size();j++)
			{
				if(i != j)
				{
					for(int k = 0;k < p[j].size();k++)
					{
						double d = pointPolygonTest(ch,Point(c1.features[p[j][k]].x,c1.features[p[j][k]].y), true);
						if(d >= -5)
						{
							countInside++;
						}
					}
				}
			}
			//cout << "End of Convex Hull " << i << " Size = " << ch.size() << endl;
		}
		c1.fit -= countInside;
	}

	void test_fit_line(Candidate&c1)
	{
		if(c1.fit > 1)
		{
			double factor = 100;
			// then calculate the angles between points and update the fitness with ratio using the standard deviation of angles
			double diff = get_angle_diff(c1.features);
			factor += (2 * (2 - diff)); 
			c1.fit *= (factor / 100);
		}
	}

	void test_face_parts_distances(Candidate&c1)
	{
		if(c1.fit > 1)
		{
			vector<double> v = get_normalized_dists(c1.features);
			double error = 0;
			for(int i = 0;i < v.size();i++)
			{
				double test = abs(v[i] - avg_dists[i]);
				if(test > var_dists[i])
				{
					error += (test - var_dists[i]);
				}
			}
			c1.fit *= (1 - error);
		}
	}

	double calc_fitness(Candidate &c1)
	{
		//puts("Calc Firtness");
		// intially the fitness is simply the summation of the responses
		c1.fit = 0;
		for(int i = 0;i < c1.features.size();i++)
		{
			c1.fit += (c1.features[i].prob);
		}
		test_convex_hull(c1);
		//test_face_parts_distances(c1);
		check_angles_90(c1);

		check_parts_one_line(c1);
		//puts("End Calc Fitness");
		return c1.fit;
	}

	// get the standard deviation between angles of the parts
	template<class T>
	double get_std_dev_parts_angles(vector<T> &c1)
	{
		vector<Point> pts = get_mean_parts_locations(c1); 
		vector<double> angles2;
		for(int i = 0;i < pts.size();i++)
			{
				for(int j = i + 1;j < pts.size();j++)
				{
					angles2.push_back(atan2((double)pts[j].y - pts[i].y,
					pts[j].x - pts[i].x));
				}
			}
		double avg2 = 0;
		double std_dev2 = 0;
		for(int i = 0;i < angles2.size();i++)
		{
			angles2[i]  = angles2[i] * 180 / PI;
			avg2 += angles2[i];
			//cout << angles2[i] << endl;
		}
		avg2 /= angles2.size();
		for(int i = 0;i < angles2.size();i++)
		{
			std_dev2 += ((angles2[i] - avg2) * (angles2[i] - avg2));
		}

		return sqrt(std_dev2);
	}

	// check that all mean face parts are on one line
	void check_parts_one_line(Candidate &c1)
	{
		if(c1.fit > 1)
		{
			double avg = 13.807; // for all images
			double std_dev = 9.13921; // for all images
			double std_dev2 = get_std_dev_parts_angles(c1.features);
			//std_dev2 = sqrt(std_dev2 / angles2.size());
			if(std_dev2 > avg + 2*std_dev)
			{
				std_dev2 -= (avg + 2*std_dev);
				c1.fit -= std_dev2;
			}
		}
	}

	// get average points between face parts
	template<class T>
	vector<Point> get_mean_parts_locations(vector<T> &v)
	{
		// get average parts
		int parts[] = {8,18,22,28,29};
		int st = 0;
		vector<Point> vp;
		for(int j = 0;j < 5;j++)
		{
			int cnt = 0;
			Point p(0,0);
			while(st < parts[j])
			{
				p.x += v[st].x;
				p.y += v[st++].y;
				cnt++;
			}
			p.x /= cnt;
			p.y /= cnt;
			vp.push_back(p);
		}
		return vp;
	}

	// get normalized distances between face parts
	template<class T>
	vector<double> get_normalized_dists(vector<T> &v)
	{
		// get average parts
		vector<Point> vp = get_mean_parts_locations(v);
		vector<double> dists;
		double maxx = 0;
		for(int j = 0;j < vp.size() - 1;j++)
		{
			for(int k = j + 1;k < vp.size();k++)
			{
				dists.push_back(cv::norm(vp[j]-vp[k]));
			}
			maxx = max(maxx, dists[dists.size() - 1]);
		}
		// normalize using the largest 
		for(int j = 0;j < dists.size();j++)
		{
			//printf("%.2llf\t", dists[j] / maxx);
			dists[j] /= maxx;
		}
		return dists;
	}

	// get all distances for fid points parts (means and variance)
	void get_normalized_dist_all()
	{
		vector<vector<double> > all_dists;
		for(int i = 0;i < all_fids.size();i++)
		{
			all_dists.push_back(get_normalized_dists(all_fids[i]));
		}
		avg_dists.clear();
		var_dists.clear();
		avg_dists.resize(10,0);
		var_dists.resize(10,0);
		// get mean
		for(int i = all_dists.size() - 1;i >= 0;i--)
		{
			for(int j = 0;j < avg_dists.size();j++)
			{
				avg_dists[j] += all_dists[i][j];
			}
		}
		for(int j = 0;j < avg_dists.size();j++)
		{
			avg_dists[j] /= all_dists.size();
		}
		// then get variance
		for(int i = all_dists.size() - 1;i >= 0;i--)
		{
			for(int j = 0;j < avg_dists.size();j++)
			{
				var_dists[j] += (all_dists[i][j] - avg_dists[j]) * (all_dists[i][j] - avg_dists[j]);
			}
		}
		for(int j = 0;j < avg_dists.size();j++)
		{
			var_dists[j] = sqrt(var_dists[j] / all_dists.size());
		}
		// then display both of them for demo only
		/*for(int i = 0;i < avg_dists.size();i++)
		{
			printf("%.2llf\t", avg_dists[i]);
		}
		puts("");
		for(int i = 0;i < var_dists.size();i++)
		{
			printf("%.2llf\t", var_dists[i]);
		}
		system("pause");*/
	}



	double calc_fitness(Candidate &c1, vector<int> &pts)
	{
		// intially the fitness is simply the summation of the responses
		double res = 0;
		for(int i = 0;i < pts.size();i++)
		{
			res += c1.features[pts[i]].prob;
		}
		return res;
	}
	Candidate run_GA(int generations = 400,double cross_over_rate = 0.5,double mutation_rate = 0.1)
	{
		double best = -1000;
		int noChangeCounter = 0;
		int nochangeCounter_2 = 0;
		double maxFit = 150;
		for(int iter = 0;(iter < generations && best < maxFit);iter++)
		{
			//cout << "Iteration " << iter << " started\n";
			vector<Candidate> offspring;
			// cross over
			for(int i = 0;i < pop.size();i++)
			{
				// get random probability
				double p = rand()%1000;
				p = p / 1000.0;
				if(p < cross_over_rate)
				{// then select pair and do cross over
					int j;
					do
					{
						j = rand()%pop.size();
					}while(i == j);
					// get the points from the first and second candiadates
					vector<DetecetedPoint> v1,v2,v3;
					//
					Candidate cd = this->cross_over(next(pop.begin(), i)->second, 
						next(pop.begin(), j)->second,v1,v2,v3);
					this->calc_fitness(cd);
					offspring.push_back(cd);
#if _VIS
					if(next(pop.begin(), i)->second.fit < (cd.fit - 5) 
						&& next(pop.begin(), j)->second.fit < (cd.fit - 5))
					{
						static int cross_over_index = 0;
						printf("%.4d Before = %llf,%llf\tAfter = %llf\n", 
							cross_over_index,next(pop.begin(), i)->second.fit,
							next(pop.begin(), j)->second.fit, cd.fit);
						//system("pause");
						// display
						ostringstream ostr1,ostr2,ostr3;
						// first image
						ostr1 << "cross_over_" << setw(4) << setfill('0') << cross_over_index;
						ostr1 << "_1_" << next(pop.begin(), i)->second.fit << "_.jpg";
						Mat tmpimg = Visualization::overlayFidsImage(this->img1,next(pop.begin(), i)->second.features,cv::Scalar(255,0,0));
						imwrite(ostr1.str().c_str(),tmpimg);
						// second image
						ostr2 << "cross_over_" << setw(4) << setfill('0') << cross_over_index;
						ostr2 << "_2_" << next(pop.begin(), j)->second.fit << "_.jpg";
						tmpimg = Visualization::overlayFidsImage(this->img1,next(pop.begin(), j)->second.features,cv::Scalar(0,0,255));
						imwrite(ostr2.str().c_str(),tmpimg);
						// result image
						ostr3 << "cross_over_" << setw(4) << setfill('0') << cross_over_index;
						ostr3 << "_3_" << cd.fit << "_.jpg";
						//cout << v1.size() << "\t" << v2.size() << endl;
						tmpimg = Visualization::overlayFidsImage(this->img1,v1,cv::Scalar(255,0,0));
						tmpimg = Visualization::overlayFidsImage(tmpimg,v2,cv::Scalar(0,0,255));
						tmpimg = Visualization::overlayFidsImage(tmpimg,v3,cv::Scalar(0,255,0));
						imwrite(ostr3.str().c_str(),tmpimg);
						// advance
						cross_over_index++;
					}
#endif
				}
			}

			// mutation
			for(int i = 0;i < pop.size();i++)
			{
				// get random probability
				double p = rand()%1000;
				p = p / 1000.0;
				if(p < mutation_rate)
				{// then select pair and do mutation
					Candidate cd = this->mutate(next(pop.begin(), i)->second);
					this->calc_fitness(cd);
					offspring.push_back(cd);
					//printf("Before = %llf\tAfter = %llf\n", next(pop.begin(), i)->second.fit, cd.fit);
					//system("pause");
				}
			}
			// Then add them to the population
			for(int i = 0;i < offspring.size();i++)
			{
				insert_candidate(offspring[i]);
				// and remove the lowest (last) element
				pop.erase(next(pop.begin(), pop.size() - 1));
			}
			//cout << "Iteration " << iter << " Best Fitness = " << pop.begin()->second.fit << endl;
			if(pop.begin()->second.fit > best)
			{
				best = pop.begin()->second.fit;
				noChangeCounter = 0;
				nochangeCounter_2 = 0;
			}
			else
			{
				// increase the counter
				if(++noChangeCounter > 10 && best < maxFit)
				{
					// then reintialize the system
					noChangeCounter = 0;
					int popSize = pop.size();
					pop.erase(next(pop.begin(),1), pop.end());
					bestResponse.clear();
					bestFidNum.clear();
					map<double, int> best_point_reponses;
					for(int i = 0;i < 29;i++)
					{
						if(bestPoints[i].prob > 0.8)
						{
							bestResponse.push_back(bestPoints[i]);
							bestFidNum.push_back(i);
						}
						else
						{
							best_point_reponses[bestPoints[i].prob] = i;
						}
					}
					map<double, int>::reverse_iterator rev_iter = best_point_reponses.rbegin();
					while(bestFidNum.size() < 2)
					{
						bestResponse.push_back(bestPoints[rev_iter->second]);
						bestFidNum.push_back(rev_iter->second);
						rev_iter++;
					}
					// and complete the population
					init(img1,popSize,false,false,false);
				}
				if(++nochangeCounter_2 > 50 && sizeFactor < 1)
				{
					nochangeCounter_2 = 0;
					cout << "Re-intialize with factor = " << sizeFactor*2 << endl;
					// double the size for more accurate results
					noChangeCounter = 0;
					iter -= 50;
					int pop_size = pop.size();
					re_init(pop_size);
				}
			}
		}
		// average with the best points
		cout << "\nAverage with the best points\n";
		Candidate cd = pop.begin()->second;
		for(int i = 0;i < 29;i++)
		{
			double len = getLength(cd.features[i].x,cd.features[i].y,bestPoints[i].x,bestPoints[i].y);
			cout << len << "\t";
			if(len < 5)
			{
				// if distance less then 1/10 of the inter-occular distance
				// so the point is near and we can make approximations
				cd.features[i].x = (bestPoints[i].x + cd.features[i].x) / 2;
				cd.features[i].y = (bestPoints[i].y + cd.features[i].y) / 2;
			}
		}
		cout << "\nBest Fitness = " << cd.fit << endl;
		// back to normal size
		for(int i = 0;i < 29;i++)
		{
			cd.features[i].x /= sizeFactor;
			cd.features[i].y /= sizeFactor;
		}
		// then display the highest reponse in the population
		//Testing::displayDetectedFidsImage(img1,cd.features);
		
		return cd;
	}

	// display labels for one of the sets (test, train)
	string testLFPW(string base,string Folder, string type, int startingIndex,int endIndex,
		double crossover = 0.5,double mutation = 0.1,int population = 30,int generations = 400)
	{
		// load faces for testing
		vector<FaceData> allFaces = Loading::loadAllFaces(base + Folder,type);
		// load all fiducial points for the training
#if TEST_PART
		all_fids = Loading::loadAllFidPoints(Folder, "test");
		//Testing::displayFidsImage(allFaces[0].img,allFaces[0].fids);
#else
		all_fids = Loading::loadAllFidPointsWithMirrors(base + Folder, "train",false);
		get_normalized_dist_all();
		div_fids.resize(1);
		div_fids[0].start = 0;
		div_fids[0].size = all_fids.size();
		this->div_ind = 0;
#endif
		// loading ranges and SVM models
		svm_poses.push_back(new MySVMS(base + "combined_models\\fid"));
		// create the output directory
		char directory[200];
		sprintf(directory,"LFPW_%.2f_%.2f_%.4d_%.4d", crossover,mutation,population,generations);
		// remove directory if exists
		/*string remove = "RMDIR ";
		remove += directory;
		remove += " /S /Q";
		system(remove.c_str());*/
		// create new directory
		for(int img_ind = startingIndex;img_ind <= endIndex;img_ind++)
		{
			//if(img_indstartingIndexndex)
			{
				vector<Fiducial> fids = allFaces[img_ind - 1].get_scaled_fids();
				// init
				time_t first,last;
				time(&first);
				// test different divisions
				calcDivIndex = false;
				
				GA::init(allFaces[img_ind - 1].get_img(),population,true,true,true);
				
				// run
				Candidate res = run_GA(generations,crossover,mutation);
				time(&last);
				double seconds = difftime(last,first);
				
				// display the labels
				//cv::Mat tempImg = Testing::overlayDetectedFidsImage(allFaces[img_ind - 1].img,res.features);
				char imgfile[100];
				sprintf(imgfile,"%s/fid%.4d.dat", directory, img_ind);
				// create the folder if doesnot exist
				struct stat st = {0};
				if (stat(directory, &st) == -1) {
    					mkdir(directory, 0700);
				}
				//mkdir(directory);
				// save the datafile
				allFaces[img_ind - 1].get_original(res.features,base + imgfile);				
				//cout << imgfile << endl;
				printf ("\nTime seconds.%s = %llf seconds\n",imgfile, seconds);
			}
		}
		// delete svms
		for(int i = 0;i < svm_poses.size();i++)
		{
			delete svm_poses[i];
		}
		return directory;
	}

	// display labels for one of the sets (test, train)
	void testHelen(string base,string Folder, string helen_folder, int startingIndex,int endIndex)
	{
		// load faces for testing
		vector<FaceData> allFaces = Loading::loadAllHelenFaces(base + helen_folder);
		// load all fiducial points for the training
		all_fids = Loading::loadAllFidPointsWithMirrors(base + Folder, "train");
		div_fids.resize(1);
		div_fids[0].start = 0;
		div_fids[0].size = all_fids.size();
		this->div_ind = 0;

		// loading ranges and SVM models
		svm_poses.push_back(new MySVMS(base + "combined_models\\fid"));
		
		for(int img_ind = startingIndex;img_ind <= endIndex;img_ind++)
		{
			//if(img_ind >= startingIndex)
			{
				vector<Fiducial> fids = allFaces[img_ind - 1].get_scaled_fids();
				// init
				time_t first,last;
				time(&first);
				// test different divisions
				calcDivIndex = false;
				GA::init(allFaces[img_ind - 1].get_img(),30,true,true,true);
				
				// run
				Candidate res = run_GA(400);
				time(&last);
				double seconds = difftime(last,first);
				
				// display the labels
				//cv::Mat tempImg = Testing::overlayDetectedFidsImage(allFaces[img_ind - 1].img,res.features);
				char imgfile[100];
				sprintf(imgfile,"helen_output/fid%.4d.dat", img_ind);
				allFaces[img_ind - 1].get_original(res.features,base + imgfile);				
				//cout << imgfile << endl;
				printf ("\nTime seconds.%s = %llf seconds\n",imgfile, seconds);
				//cv::imwrite(imgfile, tempImg);
				//// then compare
				//double inter_occular = sqrt((fids[16].x - fids[17].x) * (fids[16].x - fids[17].x)
				//	+ (fids[16].y - fids[17].y) * (fids[16].y - fids[17].y));
				//cout << "inter_occular = " << inter_occular << endl;
				//// calculate the error
				//ofstream oup_err("helen_output/errors.csv",fstream::app);
				//for(int i = 0;i < res.features.size();i++)
				//{
				//	double a = (fids[i].x - res.features[i].x) * (fids[i].x - res.features[i].x) 
				//		+ (fids[i].y - res.features[i].y) * (fids[i].y - res.features[i].y);
				//	a = sqrt(a);
				//	cout << a << "\t";
				//	a /= inter_occular;
				//	oup_err << a;
				//	if(i < res.features.size() - 1)
				//	{
				//		oup_err << ",";
				//	}
				//	else
				//	{
				//		oup_err << endl;
				//		oup_err.close();
				//	}
				//}
				//cout << endl;
			}
		}
	}

	// test BIO ID
	void testBioID(string base,string Folder, string BioID_folder, int startingIndex, int endIndex)
	{
		// load faces for testing
		vector<FaceData> allFaces = Loading::loadAllBioIDFaces(base + BioID_folder);
		// load all fiducial points for the training
		all_fids = Loading::loadAllFidPointsWithMirrors(base + Folder, "train");
		div_fids.resize(1);
		div_fids[0].start = 0;
		div_fids[0].size = all_fids.size();
		this->div_ind = 0;

		// loading ranges and SVM models
		svm_poses.push_back(new MySVMS(base + "combined_models\\fid"));

		for(int img_ind = startingIndex;img_ind <= endIndex;img_ind++)
		{
			//if(img_ind >= startingIndex)
			{
				vector<Fiducial> fids = allFaces[img_ind - 1].get_scaled_fids();
				// init
				time_t first,last;
				time(&first);
				// test different divisions
				calcDivIndex = false;
				GA::init(allFaces[img_ind - 1].get_img(),30,true,true,true);
				
				// run
				Candidate res = run_GA(300);
				time(&last);
				double seconds = difftime(last,first);
				
				// display the labels
				/*static int map_ind[] = {1,2,3,4,0,0,0,0,9,10,11,12,0,0,0,0,17,18,19,20,21,0,23,24,25,0,0,28,0};
				for(int j = 0;j < res.features.size();j++)
				{
					if(!map_ind[j])
					{
						res.features[j].x = 0;
						res.features[j].y = 0;
					}
				}
			*/
				//Mat tempImg = Testing::overlayDetectedFidsImage(allFaces[img_ind - 1].img,res.features);
				char imgfile[100];
				sprintf(imgfile,"bioid_output/fid%.4d.dat", img_ind);
				allFaces[img_ind - 1].get_original(res.features,base + imgfile);				
				//cout << imgfile << endl;
				printf ("\nTime seconds.%s = %llf seconds\n",imgfile, seconds);
				//cv::imwrite(imgfile, tempImg);
				//// then compare
				//double inter_occular = sqrt((fids[16].x - fids[17].x) * (fids[16].x - fids[17].x)
				//	+ (fids[16].y - fids[17].y) * (fids[16].y - fids[17].y));
				//cout << "inter_occular = " << inter_occular << endl;
				//// calculate the error
				//ofstream oup_err("bioid_output/errors.csv",fstream::app);
				//for(int i = 0;i < res.features.size();i++)
				//{
				//	double a = (fids[i].x - res.features[i].x) * (fids[i].x - res.features[i].x) 
				//		+ (fids[i].y - res.features[i].y) * (fids[i].y - res.features[i].y);
				//	a = sqrt(a);
				//	cout << a << "\t";
				//	a /= inter_occular;
				//	oup_err << a;
				//	if(i < res.features.size() - 1)
				//	{
				//		oup_err << ",";
				//	}
				//	else
				//	{
				//		oup_err << endl;
				//		oup_err.close();
				//	}
				//}
				//cout << endl;
			}
		}
	}
};

GA* GA::inst = 0;

#endif
