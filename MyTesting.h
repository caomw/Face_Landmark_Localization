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


#ifndef MYTESTING_H
#define MYTESTING_H

// OpenCV_Helloworld.cpp : Defines the entry point for the console application.
// Created for build/install tutorial, Microsoft Visual Studio and OpenCV 2.2.0
#include <vector>
#include <list>
#include <map>
#include <set>
#include <queue>
#include <deque>
#include <stack>
#include <bitset>
#include <algorithm>
#include <functional>
#include <numeric>
#include <utility>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <string>
#include <cstring>
#include <fstream>
//#include <Windows.h>
//#include "../libsvm-3.17/svm.h"
#include "MySift.h"
//#include "test_affine.h"
//#include "ICP.h"
#include "mylibsvm.h"
//#include "lbp.h"
#include "histogram.hpp"
#include "superpixel/segment-image.h"
#include "MyHOGDescriptor.h"
#include "MyHistograms.h"
#include "MyLoading.h"
//#include "MyLiop.h"
//#include "superpixel\SLIC.h"
#include "visualization.h"

class Testing
{
public:

	static Mat getDescriptors(Mat img,vector<KeyPoint> keypoints)
	{
		return MySIFT::getSiftDescriptors(img,keypoints);
	}

	// calculate response map for one fiducial point
	static double calcResponseOnePoint(svm_model *mymodel, vector<float>& inp, int dim, 
		double lower,double upper, vector<double>&feature_min, vector<double>& feature_max)
	{
		// classify
		svm_node x[1000]; // assuming 1000 features
		for(int j = 0;j < dim;j++)
		{
			x[j].index = j + 1;
			double value = inp[j];
			// do the scaling
			if(value == feature_min[j])
				value = lower;
			else if(value == feature_max[j])
				value = upper;
			else
				value = lower + (upper-lower) * 
						(value-feature_min[j])/
						(feature_max[j]-feature_min[j]);
			// end scaling
			x[j].value = value;
		}
		x[dim].index = -1;
		return svm_predict(mymodel,x);
	}


		// calculate response map for one fiducial point
	static double calcResponseOnePoint(svm_model *mymodel, vector<double>& inp, int dim, 
		double lower,double upper, vector<double>&feature_min, vector<double>& feature_max)
	{
		// classify
		svm_node x[1000]; // assuming 1000 features
		for(int j = 0;j < dim;j++)
		{
			x[j].index = j + 1;
			double value = inp[j];
			// do the scaling
			if(value == feature_min[j])
				value = lower;
			else if(value == feature_max[j])
				value = upper;
			else
				value = lower + (upper-lower) * 
						(value-feature_min[j])/
						(feature_max[j]-feature_min[j]);
			// end scaling
			x[j].value = value;
		}
		x[dim].index = -1;
		return svm_predict(mymodel,x);
	}


	// calculate response map for one fiducial point
	static vector<double> calcResponseMap(svm_model *mymodel, float* inp, int size, int dim, double lower,double upper, vector<double>&feature_min, vector<double>& feature_max, double &max_out,double &min_out)
	{
		max_out = -1000000;
		min_out = 10000000;
		vector<double> vec;
			// classify
		svm_node x[1000];
		for(int i = 0;i < size;i++)
		{
			for(int j = 0;j < dim;j++)
			{
				x[j].index = j + 1;
				double value = inp[i * dim + j];
				// do the scaling
				if(value == feature_min[j])
					value = lower;
				else if(value == feature_max[j])
					value = upper;
				else
					value = lower + (upper-lower) * 
							(value-feature_min[j])/
							(feature_max[j]-feature_min[j]);
				// end scaling
				x[j].value = value;
			}
			x[dim].index = -1;
			double res = svm_predict(mymodel,x);
			max_out = max(max_out, res);
			min_out = min(min_out, res);
			vec.push_back(res);
		}
		return vec;
	}

	// calculate all reponse maps for one image for one fiducial point
	static vector<double> calcAllResponseMapsForOneImageOnePoint(int fidNum,double &max_out,double &min_out,
		svm_model * loadedSVMModel,Mat descriptors,string name)
	{
		// but first load the range file to scale the data again
		double miny, maxy;
		vector<double> minv;
		vector<double> maxv;
		Loading::loadRange(miny,maxy,minv,maxv,fidNum,name);
		// calc response map
		vector<double> tmp_res_map = calcResponseMap(loadedSVMModel,(float*)((void*)descriptors.data),descriptors.rows, descriptors.cols, 
			miny, maxy, minv, maxv, max_out,min_out);

		return tmp_res_map;
	}


	// get key point
	static KeyPoint getKeyPoint(float x, float y, int angle)
	{
		KeyPoint kp;
		kp.angle = angle;
		kp.class_id = -1;
		kp.octave = 0;
		kp.pt.x = x;
		kp.pt.y = y;
		//kp.response = 0.05;
		kp.size = radius*2; // The diameter

		return kp;
	}


	// the indeces are the points used in this part only
	static vector<Fiducial> testUsingRanasac(int k,vector<vector<DetecetedPoint> > &res_maps, 
		vector<vector<Fiducial> > &all_fids,double &prob,vector<int> &indeces)
	{
		// choose two random points
		int n = 2;
	
		/// Get rotation matrix 2X2
		vector<Point2f> dstTransform(all_fids[k].size()); // the examplar transformed to the detected points
		vector<Point2f> dstPts(all_fids[k].size()); // the detected points transformed to the examplar data

		double dc_X = 0; // X DC component for detected point
		double dc_Y = 0; // Y DC component for detected point

		double model_X = 0; // X DC component for used model 
		double model_Y = 0; // Y DC component for used model 

		for(int i = 0;i < all_fids[k].size();i++)
		{
			model_X += all_fids[k][i].x; 
			model_Y += all_fids[k][i].y; 

			dstPts[i] = Point2f(0,0);
		}

		model_X /= all_fids[k].size();
		model_Y /= all_fids[k].size();

		for(int i = 0 ;i < indeces.size();i++)
		{
			// set  destination point
			DetecetedPoint dp1 = res_maps[indeces[i]][rand()%res_maps[indeces[i]].size()];
			dc_X += dp1.x; 
			dc_Y += dp1.y; 

			// copy to the required array
			dstPts[indeces[i]] = Point2f(dp1.x, dp1.y);
		}
		dc_X /= indeces.size();
		dc_Y /= indeces.size();

		vector<int> ind(n);
		for(int i = 0;i < n;i++)
		{
			bool flag = false;
			do
			{
				flag = false;
				ind[i] = rand()%indeces.size();
				for(int j = 0;j < i;j++)
					flag = flag || (ind[i] == ind[j]);
				// check if this point in the detection is less than 0.5
				if(res_maps[indeces[ind[i]]][0].prob < 1)
				{
					flag = false;
				}
			}while(flag);
		}
		//
		Mat srcX( 2, 2, CV_32FC1 );
		Mat dstB( 2, 2, CV_32FC1 );
		Mat rot_mat( 2, 2, CV_32FC1 );
		// fill the sample points
		for(int i = 0 ;i < n;i++)
		{ 
			srcX.at<float>(cv::Point(0, i)) = (all_fids[k][indeces[ind[i]]].x - model_X);
			srcX.at<float>(cv::Point(1, i)) = (all_fids[k][indeces[ind[i]]].y - model_Y);

			dstB.at<float>(cv::Point(0, i)) = (dstPts[indeces[ind[i]]].x - dc_X);
			dstB.at<float>(cv::Point(1, i)) = (dstPts[indeces[ind[i]]].y - dc_Y);
		}
		// get the transformation
		rot_mat = srcX.inv() * dstB;

		// then translate the rest of points according to this translation
		Mat X( all_fids[k].size(), 2, CV_32FC1 );
		Mat tmpB( all_fids[k].size(), 2, CV_32FC1 );
		// fill all the points
		for(int i = 0 ;i < all_fids[k].size();i++)
		{ 
			X.at<float>(cv::Point(0, i)) = (all_fids[k][i].x - model_X);
			X.at<float>(cv::Point(1, i)) = (all_fids[k][i].y - model_Y);

			tmpB.at<float>(cv::Point(0, i)) = (dstPts[i].x - dc_X);
			tmpB.at<float>(cv::Point(1, i)) = (dstPts[i].y - dc_Y);
		}
		// get the new Mat
		Mat B = X * rot_mat;
		Mat tmpX = tmpB * rot_mat.inv();

		// and copy them back to the array used in the rest of the code
		for(int i = 0 ;i < all_fids[k].size();i++)
		{ 
			dstTransform[i].x = (B.at<float>(cv::Point(0, i)) + dc_X);
			dstTransform[i].y = (B.at<float>(cv::Point(1, i)) + dc_Y);

			dstPts[i].x = (tmpX.at<float>(cv::Point(0, i)) + model_X);
			dstPts[i].y = (tmpX.at<float>(cv::Point(1, i)) + model_Y);
		}
	
		// then calculate probability
		prob = 1;
		vector<Fiducial> tmpRes(all_fids[k].size());
		double inter_occular = (all_fids[k][16].x - all_fids[k][17].x) * (all_fids[k][16].x - all_fids[k][17].x)
				+ (all_fids[k][16].y - all_fids[k][17].y) * (all_fids[k][16].y - all_fids[k][17].y);
		
		for(int i = 0;i < indeces.size();i++)
		{
			double dx = (all_fids[k][indeces[i]].x - dstPts[indeces[i]].x);
			double dy = (all_fids[k][indeces[i]].y - dstPts[indeces[i]].y);
			// using normal distribution
			double dist_squared = (dx*dx + dy*dy) / inter_occular;
		
			tmpRes[indeces[i]].var = all_fids[k][indeces[i]].var;
			double tmpProb = exp(-dist_squared / (2 * tmpRes[i].var * tmpRes[i].var));
			prob *= (tmpProb / (tmpRes[i].var * sqrt(2 * 3.14159265359)));
		}
		// set the points
		for(int i = 0;i < tmpRes.size();i++)
		{
			tmpRes[i].x = dstTransform[i].x;
			tmpRes[i].y = dstTransform[i].y;
		}
		return tmpRes;
	}




	// get fiducial points using the algorithm of David Jacobs
	static vector<Fiducial> calcFidPoints(cv::Mat &img,vector<KeyPoint> &keypoints,
		Mat &descriptors, vector<vector<Fiducial> > &all_fids, vector<svm_model*> &svm_models)
	{
		//
		vector<vector<DetecetedPoint> > res_maps(29);
		multimap<double, DetecetedPoint> all_res_maps[29];
		vector<int> indeces; // used indeces in calculations
		for(int i = 0;i < svm_models.size();i++)
		{
			// but first load the range file to scale the data again
			double max_out,min_out;
			vector<double> tmp_res_map = calcAllResponseMapsForOneImageOnePoint(i,max_out, min_out,svm_models[i],descriptors,"combined_models\\fid");
			double epsilon = 0.1;
			max_out -= epsilon;
			//svm_free_and_destroy_model(&mymodel);
			// extract the needed points
			multimap<double, DetecetedPoint> vec_points;
			vector<DetecetedPoint> bestPoints;
			double sumProb = 0;
			// print the response map
			//ostringstream response_file;
			//response_file << "D:\\response" << i << ".csv";
			//ofstream reponse(response_file.str());
			for(int j = 0;j < keypoints.size();j++)
			{
				DetecetedPoint dp;
				dp.x = keypoints[j].pt.x;
				dp.y = keypoints[j].pt.y;
				dp.prob = tmp_res_map[j];// - min_out;
				sumProb += dp.prob;
				vec_points.insert(make_pair(dp.prob,dp));
				//reponse << dp.x << "," << dp.y << "," << dp.prob << endl;
			}
			//reponse.close();
			all_res_maps[i] = vec_points;
			multimap<double, DetecetedPoint>::reverse_iterator itr = vec_points.rbegin();
			indeces.push_back(i);
			cout << "Fid Number : " << i + 1 << " best probabilities ";
			for(int j = 0;j < 2;j++, itr++)
			{
				bestPoints.push_back(itr->second);
				cout << itr->second.prob << " ";
			}
			cout << endl;
		
			res_maps[i] = bestPoints;
			// normalize the points to be as probabilities
			/*
			for(multimap<double, DetecetedPoint>::iterator j_Itr = all_res_maps[i].begin();
				j_Itr != all_res_maps[i].end();j_Itr++)
			{
				j_Itr->second.prob /= sumProb;
			}
			*/
		}
		// 
		// Ransac
		cout << "\nStarting RANSAC\n";
		// repeat for 10000 times
	
		//srand(10);
		int num_of_R = 10000;
		vector< vector<Fiducial> > best_models;
		vector<double> best_models_prob;
		for(int R = 0;R < num_of_R;R++)
		{
			// select random exemplar
			int k = rand() % all_fids.size();
			double prob;
			// then decide the best prob for each part
			for(int j = 0;j < indeces.size();j++)
			{
				vector<Fiducial> tmpRes = testUsingRanasac(k,res_maps,all_fids,prob,indeces); 
				vector<double>::iterator itr = upper_bound(best_models_prob.begin(), best_models_prob.end(), prob);
				best_models.insert(best_models.begin() + (itr - best_models_prob.begin()), tmpRes);
				best_models_prob.insert(itr, prob);
				if(best_models.size() > 100)
				{
					best_models.erase(best_models.begin());
					best_models_prob.erase(best_models_prob.begin());
				}
			}
		}
		// display best models
		/*for(int i = 0;i < best_models.size();i++)
		{
			displayFidsImage(img,best_models[i]);
		}*/
		// RANSAC finished
		cout << "RANSAC Done\n";
		// then take the best 100
		cout << "Choosing best points\n";
		vector<Fiducial> res_fin(29);
		for(int i = 0;i < res_fin.size();i++)
		{
			// get the best according to the numbers given
		
			double bestProb = 0;
			DetecetedPoint bestPoint;
			for(vector<DetecetedPoint>::iterator j_Itr = res_maps[i].begin();
				j_Itr != res_maps[i].end();j_Itr++)
			{
				double prob = 0;
			
				for(int k = 0;k < best_models.size();k++)
				{
					double dx = (j_Itr->x - best_models[k][i].x);
					double dy = (j_Itr->y - best_models[k][i].y);
					double inter_occular_2 = (best_models[k][16].x - best_models[k][17].x) * (best_models[k][16].x - best_models[k][17].x) 
						+ (best_models[k][16].y - best_models[k][17].y) * (best_models[k][16].y - best_models[k][17].y);
					// using normal distribution
					double dist_squared = (dx*dx + dy*dy) / inter_occular_2;
			
					double var = best_models[k][i].var;
					double tmpProb = exp(-dist_squared / (2 * var * var));
					prob += (tmpProb / (var * sqrt(2 * 3.14159265359)));	
				}
				prob *= j_Itr->prob;
				if(prob > bestProb)
				{
					bestProb = prob;
					bestPoint = *j_Itr;
				}
			}
		
			double minX = bestPoint.x - 10;
			double minY = bestPoint.y - 10;
			double maxX = bestPoint.x + 10;
			double maxY = bestPoint.y + 10;
			// more optimization by taking all the points in the rectangle around the best point
			for(double x = minX;x <= maxX;x++)
			{
				for(double y = minY;y <= maxY;y++)
				{
					double prob = 0;
			
					for(int k = 0;k < best_models.size();k++)
					{
						double dx = (x - best_models[k][i].x);
						double dy = (y - best_models[k][i].y);
						double inter_occular_2 = (best_models[k][16].x - best_models[k][17].x) * (best_models[k][16].x - best_models[k][17].x) 
							+ (best_models[k][16].y - best_models[k][17].y) * (best_models[k][16].y - best_models[k][17].y);
						// using normal distribution
						double dist_squared = (dx*dx + dy*dy) / inter_occular_2;
			
						double var = best_models[k][i].var;
						double tmpProb = exp(-dist_squared / (2 * var * var));
						prob += (tmpProb / (var * sqrt(2 * 3.14159265359)));	
					}
					//prob *= j_Itr->second.prob;
					if(prob > bestProb)
					{
						bestProb = prob;
						bestPoint.x = x;
						bestPoint.y = y;
					}
				}
			}
		
			res_fin[i].x = bestPoint.x;
			res_fin[i].y = bestPoint.y;

		}
	
		cout << "Final Decision\n";
		return res_fin;
	}


	// get fiducial points using the algorithm of David Jacobs
	// the indeces input are used to as the points to be tested using the Examplar based method only
	static vector<Fiducial> calcFidPoints(cv::Mat &img,vector<KeyPoint> &keypoints,
		Mat &descriptors, vector<vector<Fiducial> > &all_fids, vector<svm_model*> &svm_models,
		vector<int> indeces)
	{
		vector<vector<DetecetedPoint> > res_maps(29);
		multimap<double, DetecetedPoint> all_res_maps[29];
		for(int ii = 0;ii < indeces.size();ii++)
		{
			int fidNum = indeces[ii];
			// but first load the range file to scale the data again
			double miny, maxy;
			vector<double> minv;
			vector<double> maxv;
			Loading::loadRange(miny,maxy,minv,maxv,fidNum);
			// calc response map
			cout << " " << fidNum;
			double max_out,min_out;
			vector<double> tmp_res_map = calcResponseMap(svm_models[fidNum],(float*)((void*)descriptors.data),descriptors.rows, descriptors.cols, 
				miny, maxy, minv, maxv, max_out,min_out);
			double epsilon = 0.1;
			max_out -= epsilon;
			//svm_free_and_destroy_model(&mymodel);
			// extract the needed points
			multimap<double, DetecetedPoint> vec_points;
			vector<DetecetedPoint> bestPoints;
			double sumProb = 0;
			//puts("here!");
			for(int j = 0;j < keypoints.size();j++)
			{
				DetecetedPoint dp;
				dp.x = keypoints[j].pt.x;
				dp.y = keypoints[j].pt.y;
				dp.prob = tmp_res_map[j] - min_out;
				sumProb += dp.prob;
				vec_points.insert(make_pair(dp.prob,dp));
			}
			all_res_maps[fidNum] = vec_points;
			multimap<double, DetecetedPoint>::reverse_iterator itr = vec_points.rbegin();
		
			for(int j = 0;j < 2;j++, itr++)
			{
				bestPoints.push_back(itr->second);
			}
		
			res_maps[fidNum] = bestPoints;
			// normalize the points to be as probabilities
			for(multimap<double, DetecetedPoint>::iterator j_Itr = all_res_maps[fidNum].begin();
				j_Itr != all_res_maps[fidNum].end();j_Itr++)
			{
				j_Itr->second.prob /= sumProb;
			}

			//puts("here again");
		}
		// 
		// Ransac
		cout << "\nStarting RANSAC\n";
		// repeat for 10000 times
		int num_of_R = 10000;
		vector< vector<Fiducial> > best_models;
		vector<double> best_models_prob;
		for(int R = 0;R < num_of_R;R++)
		{
			// select random exemplar
			int k = rand() % all_fids.size();
			double prob;
			// then decide the best prob for each part
			vector<Fiducial> tmpRes = testUsingRanasac(k,res_maps,all_fids,prob,indeces); 

			vector<double>::iterator itr = upper_bound(best_models_prob.begin(), best_models_prob.end(), prob);
			best_models.insert(best_models.begin() + (itr - best_models_prob.begin()), tmpRes);
			best_models_prob.insert(itr, prob);
			if(best_models.size() > 100)
			{
				best_models.erase(best_models.begin());
				best_models_prob.erase(best_models_prob.begin());
			}
		}

		// RANSAC finished
		cout << "RANSAC Done\n";
		// then take the best 100
		vector<Fiducial> res_fin(res_maps.size());
		// get the average for the points outside indeces
		/*cout << "calculating average locations\n";
	
		for(int i = 0;i < res_fin.size();i++)
		{
			res_fin[i].x = res_fin[i].y = 0;
			for(int j = 0;j < best_models.size();j++)
			{
				//displayFidsImage(img,best_models[j]);
				res_fin[i].x += best_models[j][i].x;
				res_fin[i].y += best_models[j][i].y;
			}
			res_fin[i].x /= best_models.size();
			res_fin[i].y /= best_models.size();
		}
		*/
		cout << "Choosing best points\n";
		for(int index = 0;index < indeces.size();index++)
		{
			// get the best according to the numbers given
			int i = indeces[index];
			double bestProb = 0;
			DetecetedPoint bestPoint;
			for(vector<DetecetedPoint>::iterator j_Itr = res_maps[i].begin();
				j_Itr != res_maps[i].end();j_Itr++)
			{
				double prob = 0;
			
				for(int k = 0;k < best_models.size();k++)
				{
					double dx = (j_Itr->x - best_models[k][i].x);
					double dy = (j_Itr->y - best_models[k][i].y);
					double inter_occular_2 = (best_models[k][16].x - best_models[k][17].x) * (best_models[k][16].x - best_models[k][17].x) 
						+ (best_models[k][16].y - best_models[k][17].y) * (best_models[k][16].y - best_models[k][17].y);
					// using normal distribution
					double dist_squared = (dx*dx + dy*dy) / inter_occular_2;
			
					double var = best_models[k][i].var;
					double tmpProb = exp(-dist_squared / (2 * var * var));
					prob += (tmpProb / (var * sqrt(2 * 3.14159265359)));	
				}
				prob *= j_Itr->prob;
				if(prob > bestProb)
				{
					bestProb = prob;
					bestPoint.x = j_Itr->x;
					bestPoint.y = j_Itr->y;
				}
			}
			res_fin[i].x = bestPoint.x;
			res_fin[i].y = bestPoint.y;
		}
	
		cout << "Final Decision\n";
		return res_fin;
	}


	static vector<Fiducial> calcAllResponseMaps(cv::Mat &dst, vector<vector<Fiducial> > &all_fids,vector<svm_model*> &svm_models)
	{
		// the sliding window
		vector<KeyPoint> keypoints;
		// each two consecutive descriptors will be one sample for the SVM
		// the step is 4 if the width is 200
		int y_step = dst.rows / 50;
		int x_step = dst.cols / 50;
		vector<vector<DetecetedPoint> > res_map;
		for(int y = 4;y < dst.rows - 4;y+=y_step)
		{
			vector<DetecetedPoint> vdp;
			DetecetedPoint dp;
			dp.y = y;
			for(int x = 4;x < dst.cols - 4;x+=x_step)
			{
				dp.x = x;
				vdp.push_back(dp);
				keypoints.push_back(getKeyPoint(x,y,0));
			}
			res_map.push_back(vdp);
		}
		
		// then extract all descriptors to be used 
		// in creating the response map for each fiducial point
		Mat descriptors = getDescriptors(dst,keypoints);
		// calculate the points locations
		return calcFidPoints(dst, keypoints, descriptors,all_fids,svm_models);
	}

	// calculate the 
	static vector<Fiducial> calcAllResponseMaps(cv::Mat &dst, vector<vector<Fiducial> > &all_fids,
		vector<svm_model*> &svm_models,vector<int> indeces, int randomPoints = 0)
	{
		// the sliding window
		vector<KeyPoint> keypoints;
		// each two consecutive descriptors will be one sample for the SVM
		// the step is 4 if the width is 200
		if(!randomPoints)
		{
			int y_step = max(dst.rows / 100, 1);
			int x_step = dst.cols / 50;
			for(int y = 4;y < dst.rows*2 / 5;y++)
			{
				for(int x = 4;x < dst.cols - 4;x++)
				{
					keypoints.push_back(getKeyPoint(x,y,0));
				}
			}
		}
		else
		{
			vector<KeyPoint> tempkeypoints;
			for(int i = 0;i < randomPoints;i++)
			{
				tempkeypoints.push_back(getKeyPoint(4 + (rand() % (dst.cols - 8)),4 + (rand() % (dst.rows - 8)),0));
			}
			Mat tempdescriptors = getDescriptors(dst,tempkeypoints);
			vector<Fiducial> temp = calcFidPoints(dst, tempkeypoints, tempdescriptors,all_fids,svm_models);
			// get the new dimensions
		
			for(int i = 0;i < indeces.size();i++)
			{
				double minx = temp[indeces[i]].x,miny = temp[indeces[i]].y;
				double maxx = temp[indeces[i]].x,maxy = temp[indeces[i]].y;
				minx = max(minx - 2, 0.0);
				miny = max(miny - 2, 0.0);
				maxx = min(maxx + 2, dst.cols - 1.0);
				maxy = min(maxy + 2, dst.rows - 1.0);
				for(int y = miny;y <= maxy;y++)
				{
					for(int x = minx;x <= maxx;x++)
					{
						keypoints.push_back(getKeyPoint(x,y,0));
					}
				}
			}
		}
		// then extract all descriptors to be used 
		// in creating the response map for each fiducial point
		Mat descriptors = getDescriptors(dst,keypoints);
		// calculate the points locations
		return calcFidPoints(dst, keypoints, descriptors,all_fids,svm_models,indeces);
	}


	static vector<Fiducial> calcAllResponseMaps2(cv::Mat &dst, vector<vector<Fiducial> > &all_fids,
		vector<svm_model*> &svm_models,vector<int> indeces, int randomPoints)
	{
		// the sliding window
		vector<KeyPoint> keypoints;
		vector<KeyPoint> tempkeypoints;
		for(int i = 0;i < randomPoints;i++)
		{
			tempkeypoints.push_back(getKeyPoint(4 + (rand() % (dst.cols - 8)),4 + (rand() % (dst.rows - 8)),0));
		}
		Mat tempdescriptors = getDescriptors(dst,tempkeypoints);
		vector<Fiducial> temp = calcFidPoints(dst, tempkeypoints, tempdescriptors,all_fids,svm_models);
		// get the new dimensions
		
		for(int i = 0;i < indeces.size();i++)
		{
			double minx = temp[indeces[i]].x,miny = temp[indeces[i]].y;
			double maxx = temp[indeces[i]].x,maxy = temp[indeces[i]].y;
			minx = max(minx - 2, 0.0);
			miny = max(miny - 2, 0.0);
			maxx = min(maxx + 2, dst.cols - 1.0);
			maxy = min(maxy + 2, dst.rows - 1.0);
			for(int y = miny;y <= maxy;y++)
			{
				for(int x = minx;x <= maxx;x++)
				{
					keypoints.push_back(getKeyPoint(x,y,0));
				}
			}
		}
		// then extract all descriptors to be used 
		// in creating the response map for each fiducial point
		Mat descriptors = getDescriptors(dst,keypoints);
		// calculate the points locations
		return calcFidPoints(dst, keypoints, descriptors,all_fids,svm_models,indeces);
	}


	// display labels for one of the sets (test, train)
	static void testFids(string Folder, string type, int startingIndex = 1)
	{
		// load all fiducial points for the training
		vector<vector<Fiducial> > all_fids = Loading::loadAllFidPoints(Folder, "train");
		vector<svm_model*> svm_models = Loading::loadSVMModels("combined_models/fid");
		vector<FaceData> allFaces = Loading::loadAllFaces(Folder,type);
		for(int img_ind = 1;img_ind <= allFaces.size();img_ind++)
		{
			try
			{
				if(img_ind >= startingIndex)
				{
					time_t first,last;
					time(&first);

					Mat dst = allFaces[img_ind - 1].get_img();
					//
					vector<Fiducial> fids = allFaces[img_ind - 1].get_scaled_fids();
					// the sliding window
					vector<Fiducial> res = calcAllResponseMaps(dst, all_fids,svm_models);
					time(&last);
					double seconds = difftime(last,first);
					// display the labels
					//cv::Mat tempImg = overlayFidsImage(dst,res);
					char imgfile[100];
					sprintf(imgfile,"output_LFPW_hard/fid%.4d.dat", img_ind);
					allFaces[img_ind - 1].get_original(res,imgfile);
					printf ("\nTime seconds.%s = %llf seconds\n",imgfile, seconds);
					//cout << imgfile << endl;
					//cv::imwrite(imgfile, tempImg);
					//// then compare
					//double inter_occular = sqrt((fids[16].x - fids[17].x) * (fids[16].x - fids[17].x)
					//	+ (fids[16].y - fids[17].y) * (fids[16].y - fids[17].y));
					//cout << "inter_occular = " << inter_occular << endl;
					//// calculate the error
					//ofstream oup_err("output/errors.csv",fstream::app);
					//for(int i = 0;i < res.size();i++)
					//{
					//	double a = (fids[i].x - res[i].x) * (fids[i].x - res[i].x) 
					//		+ (fids[i].y - res[i].y) * (fids[i].y - res[i].y);
					//	a = sqrt(a);
					//	cout << a << "\t";
					//	a /= inter_occular;
					//	oup_err << a;
					//	if(i < res.size() - 1)
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
			catch(...)
			{
				cout << img_ind << endl;
				cout << "This image is corrupted or doesn't exist\n";
				return;
			}
		
		}
	}

	/** @function detectAndDisplay */
	static vector<Rect> detectAndDisplay( Mat &frame , bool display = true)
	{
		std::vector<Rect> faces;
		Mat frame_gray;
		if(frame.cols > 800)
		{
			cv::resize(frame, frame, cv::Size(800, (frame.rows * 800) / frame.cols));
		}
		cvtColor( frame, frame_gray, CV_BGR2GRAY );
	
		equalizeHist( frame_gray, frame_gray );
		string opencv = "D:/Personal/Work/Masters/person_reidentification/openCV_2_4_8/opencv";
		CascadeClassifier face_cascade(opencv + "/sources/data/haarcascades/haarcascade_frontalface_alt.xml");
	
		//-- Detect faces
		face_cascade.detectMultiScale( frame_gray, faces);

		for( int i = 0; i < faces.size(); i++ )
		{
			Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
			ellipse( frame_gray, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
		}
		//-- Show what you got
		if(display)
		{
			imshow( "window_name", frame_gray );
			cvWaitKey(0);
			cvDestroyWindow("window_name");
		}
		return faces;
	}

	// return the detected faces
	static vector<Mat> testFidsOneImage(vector<svm_model*> &svm_models,
		vector<vector<Fiducial> > &all_fids, string input_img, bool display = true, bool testBlind = false)
	{
		vector<Mat> result;
		try
		{
			cv::Mat img = cv::imread(input_img.c_str());
			// get the face only
			int dx = img.cols,dy = img.rows;
			Mat dst;
			vector<Rect> faces;
			if(!testBlind)
			{
				faces = detectAndDisplay(img,display);
			}

			if((faces.size() == 0) || testBlind)
			{
				cout << "No faces detected! will search in the whole image!";
				dst = img;
				/*if(dst.cols > 400)
				{
					double newX = 200;
					cv::resize(dst,dst,cvSize(newX,(dy*newX)/dx));
				}*/
				vector<Fiducial> res = calcAllResponseMaps(dst, all_fids,svm_models);
				// display the labels
				cv::Mat tempImg = dst.clone();
				for(int j = 0;j < res.size();j++)
				{
					cv::circle(tempImg,cv::Point(res[j].x,res[j].y),1,cv::Scalar(0,0,255),1);
				}
				if(display)
				{
					imshow("Fid Points", tempImg);
					cvWaitKey(0);
				}
				result.push_back(tempImg);
			}
			else
			{
				for(int i = 0;i < faces.size();i++)
				{
					int minx = faces[i].x;
					int miny = faces[i].y;
					int maxx = faces[i].x + faces[i].width;
					int maxy = faces[i].y + (faces[i].height*1.5);

					getFaceRectangle(img,minx,miny,maxx,maxy,dx,dy);
					dst = img(cvRect(minx,miny,dx,dy));
				
					double newX = 200;
					if(dst.cols > 200)
					{
						cv::resize(dst,dst,cvSize(newX,(dy*newX)/dx));
					}
					int index[] = {8,9,10,11,12,13,14,15,16,17};
					vector<int> indeces;
					indeces.insert(indeces.begin(), index, index + 10);
					vector<Fiducial> res = calcAllResponseMaps(dst, all_fids,svm_models,indeces,300);
					// display the labels
					cv::Mat tempImg = dst.clone();
					for(int j = 0;j < res.size();j++)
					{
						cv::circle(tempImg,cv::Point(res[j].x,res[j].y),1,cv::Scalar(0,0,255),1);
					}
					if(display)
					{
						imshow("Fid Points", tempImg);
						cvWaitKey(0);
					}
					result.push_back(tempImg);
				}
			}
		}
		catch(...)
		{
			//correct << "0\n";
			cout << "This image is corrupted or doesn't exist\n";
		}
		return result;
	}

	static void testBioID(string folder)
	{
		// get all fid points
		vector<vector<Fiducial> > all_fids = Loading::loadAllFidPoints("D:\\Personal\\Work\\Masters\\lfw\\", "train");
		// get the SVM models
		vector<svm_model*> svm_models = Loading::loadSVMModels();
	
		// reading the image file name
	
		vector<string> vec;
		int start = 871;
		for(int n = start;;n++)
		{
			char filename[100];
			sprintf(filename, "%sBioID_%.4d.pgm", folder.c_str(), n);
			ifstream ifile(filename);
			if(!ifile)
			{
				break;
			}
			else
			{
				vec.push_back(filename);
			}
		}

		//#pragma omp parallel for
		for(int i = 0;i < vec.size();i++)
		{
			int n = i + start;
			cout << vec[i] << endl;
			vector<Mat> vec1 = testFidsOneImage(svm_models,all_fids, vec[i], false);
			char newfilename[100];
			sprintf(newfilename, "test_BioID\\BioID_%.4d.jpg", n);
			cout << newfilename << endl;
			imwrite(newfilename,vec1[0]);
		}
	}

	// get the maximum response points for certain fiducial point
	static DetecetedPoint detectPointsHillClimbing(int fidNum,Mat& img,vector<svm_model*>& svm_models,vector<vector<float> > &descriptors,DetecetedPoint best)
	{
		//vector<DetecetedPoint> ret;
		// convert image to gray
		/// Convert it to gray
		Mat gray;
		cvtColor( img, gray, CV_RGB2GRAY );
		gray.convertTo(gray,CV_32FC1);
		// load range for this Fid Point
		double miny, maxy;
		vector<double> minv;
		vector<double> maxv;
		Loading::loadRange(miny,maxy,minv,maxv,fidNum);
		// get the best point
		for(int i = 0;i < 100 && best.prob <= 0;i++)
		{
			best.x = rand()%img.cols;
			best.y = rand()%img.rows;
			if(descriptors[best.y * img.cols + best.x].size() == 0)
			{
				// calculate descriptor
				descriptors[best.y * img.cols + best.x] = MySIFT::getSiftDescriptor(gray, getKeyPoint(best.x,best.y,0));
			}
			// calculate the reponse
			best.prob = calcResponseOnePoint(svm_models[fidNum],descriptors[best.y * img.cols + best.x],
				minv.size(),miny,maxy,minv,maxv);
		}
	
		if(best.prob > 0)
		{
			for(int i = 0;i < 20 && best.prob < 2;i++)
			{
				// just random climbing with decreasing step
				int sx = (rand() % 10) * (((rand()&1)<<1) - 1);
				int sy = (rand() % 10) * (((rand()&1)<<1) - 1);
				//
				double y = min(max(sy + best.y,0.0),img.rows - 1.0);
				double x = min(max(sx + best.x,0.0),img.cols - 1.0);
				if(descriptors[y * img.cols + x].size() == 0)
				{
					// calculate descriptor
					descriptors[y * img.cols + x] = MySIFT::getSiftDescriptor(gray, getKeyPoint(x,y,0));
				}
				// calculate the reponse
				double prob = calcResponseOnePoint(svm_models[fidNum],descriptors[y * img.cols + x],
					minv.size(),miny,maxy,minv,maxv);

				if(prob > best.prob)
				{
					best.x = x;
					best.y = y;
					best.prob = prob;
				}
			}
		}
		else
		{
			best.x = best.y = 0;
		}
		return best;
	}

	// display labels for one of the sets (test, train)
	static void testFidsHillClimbing(string Folder, string type, int startingIndex = 1)
	{
		// load all fiducial points for the training
		vector<vector<Fiducial> > all_fids = Loading::loadAllFidPoints(Folder, "train");
		vector<svm_model*> svm_models = Loading::loadSVMModels();
		vector<FaceData> allFaces = Loading::loadAllFaces(Folder,type);
		for(int img_ind = 1;img_ind <= allFaces.size();img_ind++)
		{
			if(img_ind >= startingIndex)
			{
				// image
				Mat img = allFaces[img_ind - 1].get_img();
				// image size
				int size = img.cols * img.rows;
				// descriptors
				vector<vector<float> > descriptors(size);
				// detect points hill climbing
				vector<Fiducial> det(svm_models.size());
				/*
				for(int fid = 0;fid < svm_models.size();fid++)
				{
					DetecetedPoint p;
					det[fid] = detectPointsHillClimbing(fid,img,svm_models,descriptors,p);
				}*/
				vector<int> indeces;
				for(int k = 0;k < 29;k++) indeces.push_back(k);
				det = calcAllResponseMaps2(img,all_fids,svm_models,indeces,200);
				// display
				Visualization::displayFidsImage(img,det);
			}
		}
	}
};

#endif
