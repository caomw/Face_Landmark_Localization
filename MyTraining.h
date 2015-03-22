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


#include "MyTesting.h"

class Training
{
public:
		// train classifier for one fiducial point with near negative points
	// assuming point numbers is zero based
	static int getNegativePoint(double start, int len, double pt)
	{
		int x = (rand() % (len - 20)) + 10 + start;
		while(abs(x - pt) < 4) x = (rand() % (len - 20)) + 10 + start;

		return x;
	}
	
	static void trainFidPointNear(string Folder, string type, int pointNum)
	{
		vector< vector<float> > pos , neg;
		vector<FaceData> allFaces = Loading::loadAllFacesSorted(Folder,type);
		for(int l = 0;l < allFaces.size();l++)
		{
			Mat img = allFaces[l].get_img();
			vector<Fiducial> fids = allFaces[l].get_scaled_fids();
			//Mat descriptors;
			vector<KeyPoint> keypoints;
			// 3 positive samples
			for(int i = 0;i < 3;i++)
			{
				int mod = 10;
				int distance = 4;
				int size = 1;
				int ang = (rand() % 40) - 20;
				keypoints.push_back(Testing::getKeyPoint(fids[pointNum].x,fids[pointNum].y,ang));
				// 3 negative samples
			
				int sign = ((rand()&1)<<1) - 1;
				int x = sign * ((rand() % mod) + distance);
				while((x + fids[pointNum].x) >= img.cols || (x + fids[pointNum].x) < 0)
				{
					sign = ((rand()&1)<<1) - 1;
					x = sign * ((rand() % mod) + distance);
				}
				sign = ((rand()&1)<<1) - 1;
				int y = sign * ((rand() % mod) + distance);
				while((y + fids[pointNum].y) >= img.rows || (y + fids[pointNum].y) < 0)
				{
					sign = ((rand()&1)<<1) - 1;
					y = sign * ((rand() % mod) + distance);
				}
				keypoints.push_back(Testing::getKeyPoint(x + fids[pointNum].x,y + fids[pointNum].y,0));
			}
			try
			{
				Mat descriptors = Testing::getDescriptors(img,keypoints);
			
		
				// then fill positive and negative sample arrays
				for(int i = 0;i < 3;i++)
				{				
					vector <float> pp,nn;
					// descriptor at size 7
					for(int c = 0; c < descriptors.cols;c++)
					{
						pp.push_back(descriptors.at<float>(cv::Point(c,i*2)));
						nn.push_back(descriptors.at<float>(cv::Point(c,i*2 + 1)));
					}
					// add to the vector used in training
					pos.push_back(pp);
					neg.push_back(nn);
				}
			}
			catch(...)
			{
				//correct << "0\n";
				cout << l << endl;
				cout << "This image is corrupted or doesn't exist\n";
			}
		}
	
		// first choose parameters for training
		char ff[20];
		sprintf(ff,"fid_near%.2d", pointNum);
		ofstream trainf(ff);
		for(int i = 0;i < pos.size();i++)
		{
			trainf << "+1";
			for(int j = 0;j < pos[i].size();j++)
			{
				trainf << " " << j + 1 << ":" << pos[i][j];
			}
			trainf << endl;
		}
		for(int i = 0;i < neg.size();i++)
		{
			trainf << "-1";
			for(int j = 0;j < neg[i].size();j++)
			{
				trainf << " " << j + 1 << ":" << neg[i][j];
			}
			trainf << endl;
		}
		trainf.close();
		// then start training
		train_with_cross_validation(ff);
	}

	// train classifier for one fiducial point
	// assuming point numbers is zero based
	static void trainFidPoint(vector<FaceData>& allFaces, int pointNum,string name)
	{
		vector< vector<float> > pos , neg;
		for(int l = 0;l < allFaces.size();l++)
		{
			printf(".");
			FaceData* fd = &allFaces[l];
			vector<Fiducial> fids = fd->get_scaled_fids();
			//Mat descriptors;
			vector<KeyPoint> keypoints;
			// 3 positive samples
			Mat tmpimg = fd->get_img();
			for(int i = 0;i < 3;i++)
			{
				int ang = (rand() % 40) - 20;
				keypoints.push_back(Testing::getKeyPoint(fids[pointNum].x,fids[pointNum].y,ang));
				// 3 negative samples
				ang = (rand() % 41) - 20;
				int x = getNegativePoint(fd->rect.x, fd->rect.width,fids[pointNum].x);
				int y = getNegativePoint(fd->rect.y, fd->rect.height,fids[pointNum].y);
				keypoints.push_back(Testing::getKeyPoint(x,y,ang));
				// draw positive
				tmpimg = Visualization::drawKeyPoint(tmpimg,keypoints[i*2],cv::Scalar((i > 0)*255,(i%2)*255,0));
				// negative
				tmpimg = Visualization::drawKeyPoint(tmpimg,keypoints[i*2 + 1],cv::Scalar(0,0,255));
			}
			// save
			ostringstream oup;
			oup << "orig_" <<setw(4) << setfill('0') << l << ".jpg";
			imwrite(oup.str().c_str(),tmpimg);
			//cvWaitKey(0);
			try
			{
				Mat descriptors = Testing::getDescriptors(fd->get_img(),keypoints);
			
				// then fill positive and negative sample arrays
				for(int i = 0;i < 3;i++)
				{				
					vector <float> pp,nn;
					// descriptor at size 7
					for(int c = 0; c < descriptors.cols;c++)
					{
						pp.push_back(descriptors.at<float>(cv::Point(c,i*2)));
						nn.push_back(descriptors.at<float>(cv::Point(c,i*2 + 1)));
					}
					// add to the vector used in training
					pos.push_back(pp);
					neg.push_back(nn);
				}
			}
			catch(...)
			{
				//correct << "0\n";
				cout << l << endl;
				cout << "This image is corrupted or doesn't exist\n";
			}
		}
		// 
		puts(".");
		// first choose parameters for training
		char ff[20];
		sprintf(ff,"%s%.2d",name.c_str(), pointNum);
		ofstream trainf(ff);
		for(int i = 0;i < pos.size();i++)
		{
			trainf << "+1";
			for(int j = 0;j < pos[i].size();j++)
			{
				trainf << " " << j + 1 << ":" << pos[i][j];
			}
			trainf << endl;
		}
		for(int i = 0;i < neg.size();i++)
		{
			trainf << "-1";
			for(int j = 0;j < neg[i].size();j++)
			{
				trainf << " " << j + 1 << ":" << neg[i][j];
			}
			trainf << endl;
		}
		trainf.close();
		// then start training
		train_with_cross_validation(ff);
	}

	// assuing there is an initial training and a model already there
	static void trainFidPointWithHardNegative(vector<FaceData>& allFaces, int pointNum,string name)
	{
		// load the SVM Model
		svm_model* tempModel = Loading::loadSVMModelNumber(pointNum,name); // and don't forget to release it

		vector< vector<float> > pos , neg;
		for(int l = 0;l < allFaces.size();l++)
		{
			printf(".");
			FaceData*fd = &allFaces[l];
			vector<Fiducial> fids = fd->get_scaled_fids();
			//Mat descriptors;
			vector<KeyPoint> keypoints;
			// 3 positive samples
			Mat tmpimg = fd->get_img();
			for(int i = 0;i < 3;i++)
			{
				int ang = (rand() % 40) - 20;
				keypoints.push_back(Testing::getKeyPoint(fids[pointNum].x,fids[pointNum].y,ang));
				// draw positive
				tmpimg = Visualization::drawKeyPoint(tmpimg,keypoints[i],cv::Scalar((i > 0)*255,(i%2)*255,0));
			}
			// 3 negative samples to be the negative points that give the highest score
			// the sliding window
			vector<KeyPoint> tempKeyPoints;
			// each two consecutive descriptors will be one sample for the SVM
			// the step is 4 if the width is 200
			for(int ii = 0;ii < 100;ii++)
			{
				//negative samples
				double ang = (rand() % 41) - 20;
				int x = getNegativePoint(fd->rect.x, fd->rect.width,fids[pointNum].x);
				int y = getNegativePoint(fd->rect.y, fd->rect.height,fids[pointNum].y);
				tempKeyPoints.push_back(Testing::getKeyPoint(x,y,ang));
			}
			// then calculate the response map
			double maxOut, minOut;
			vector<double> responses = 
				Testing::calcAllResponseMapsForOneImageOnePoint(
				pointNum,
				maxOut,
				minOut,
				tempModel,
				Testing::getDescriptors(fd->get_img(),tempKeyPoints),name);
			// then get the highst false positive responses
			multimap<double, KeyPoint> tempSort;
			for(int i = 0;i < responses.size();i++)
			{
				tempSort.insert(make_pair(responses[i] - minOut,tempKeyPoints[i]));
			}
			// then get the highest responses
			multimap<double, KeyPoint>::reverse_iterator revItr = tempSort.rbegin();
			for(;revItr != tempSort.rend() && keypoints.size() < 6;revItr++)
			{
				double dx = revItr->second.pt.x - fids[pointNum].x;
				double dy = revItr->second.pt.y - fids[pointNum].y;
				keypoints.push_back(Testing::getKeyPoint(revItr->second.pt.x,revItr->second.pt.y,0));
				// negative
				tmpimg = Visualization::drawKeyPoint(tmpimg,keypoints[keypoints.size() - 1],cv::Scalar(0,0,255));
			}
			// save
#if _VIS
			ostringstream oup;
			oup << "hard_" <<setw(4) << setfill('0') << l << ".jpg";
			imwrite(oup.str().c_str(),tmpimg);
#endif
			try
			{
			
				// 
				Mat descriptors = Testing::getDescriptors(fd->get_img(),keypoints);
			
				// then fill positive and negative sample arrays
				for(int i = 0;i < 3;i++)
				{				
					vector <float> pp,nn;
					// descriptor at size 7
					for(int c = 0; c < descriptors.cols;c++)
					{
						pp.push_back(descriptors.at<float>(cv::Point(c,i)));
						nn.push_back(descriptors.at<float>(cv::Point(c,i + 3)));
					}
					// add to the vector used in training
					pos.push_back(pp);
					neg.push_back(nn);
				}
			}
			catch(...)
			{
				//correct << "0\n";
				cout << l << endl;
				cout << "This image is corrupted or doesn't exist\n";
			}
		}
		puts(".");
		// first choose parameters for training
		char ff[20];
		sprintf(ff,"%s%.2d", name.c_str(),pointNum);
		ofstream trainf(ff);
		for(int i = 0;i < pos.size();i++)
		{
			trainf << "+1";
			for(int j = 0;j < pos[i].size();j++)
			{
				trainf << " " << j + 1 << ":" << pos[i][j];
			}
			trainf << endl;
		}
		for(int i = 0;i < neg.size();i++)
		{
			trainf << "-1";
			for(int j = 0;j < neg[i].size();j++)
			{
				trainf << " " << j + 1 << ":" << neg[i][j];
			}
			trainf << endl;
		}
		trainf.close();
		// then start training
		train_with_cross_validation(ff);
	}

	static void trainAllFidPoints(string Folder, string type)
	{
		// most accurate eye points 9(8 here in zero-based) , 18 (17 here)
		//
		vector<FaceData> allFaces = Loading::loadAllFacesWithMirrorsSorted(Folder,type,false);
		
		//#pragma omp parallel for
		for(int i = 0;i < 1;i++)
		{
			for(int fid = 0 + i;fid < 2;fid+=2)
			{
				cout << "starting fid " << fid << endl;
				//trainFidPointNear(Folder,type,fid);
				string name = "temp";
				trainFidPoint(allFaces, fid,name);
				trainFidPointWithHardNegative(allFaces, fid,name) ;
			}
		}
	}

	static void trainAllFidPoints(string Folder, string type,int argc, char*argv[])
	{
		// most accurate eye points 9(8 here in zero-based) , 18 (17 here)
		//#pragma omp parallel for
		vector<FaceData> allFaces = Loading::loadAllFacesWithMirrorsSorted(Folder,type,false);
		#pragma omp parallel for
		for(int i = 1;i < argc;i++)
		{
			int fid;
			sscanf(argv[i],"%d", &fid);
			trainFidPoint(allFaces, fid,"fid");
			trainFidPointWithHardNegative(allFaces,fid,"fid");
		}
	}

	static void trainAllFidPoints_3_parts(string Folder, string type)
	{
		// most accurate eye points 9(8 here in zero-based) , 18 (17 here)
		vector<FaceData> allFaces = Loading::loadAllFacesWithMirrorsSorted(Folder,type,false);
		int parts = 3;

		#pragma omp parallel for
		for(int i = 1;i <= parts;i++)
		{
			train_part(allFaces,i,parts,"fid_part");
		}
	}

	// train this part of faces
	static void train_part(vector<FaceData>& allFaces,int partNum,int numOfParts,string name)
	{
		vector<FaceData> part;
		int size = allFaces.size() / numOfParts;
		if(partNum != numOfParts)
		{
			part.insert(part.begin(), 
				next(allFaces.begin(), (partNum - 1) * size),
				next(allFaces.begin(), partNum * size));
		}
		else
		{
			part.insert(part.begin(), 
				next(allFaces.begin(), (partNum - 1) * size),
				allFaces.end());
		}
		ostringstream ost;
		ost << name << "_" << partNum << "_" << numOfParts << "_";
		for(int fid = 0;fid < 29;fid++)
		{
			trainFidPoint(part, fid,ost.str());
			trainFidPointWithHardNegative(part, fid,ost.str());
		}
	}

	// calculate angles inside the face parts
	static double getLength(Point2f& pt)
	{
		return sqrt(pt.y*pt.y + pt.x*pt.x);
	}

	static void normalizePoint(Point2f& pt)
	{
		double len = getLength(pt);
		pt.x /= len;
		pt.y /= len;
	}

	static double calculatAngle(Point2f pt1,Point2f pt2,Point2f pt3)
	{
		double res = 0;
		pt1 -= pt2;
		pt3 -= pt2;
		// normalize
		normalizePoint(pt1);
		normalizePoint(pt3);
		// get the rotation angle
		double theta = atan2(pt3.y, pt3.x);
		double X = pt1.x * cos(theta) - pt1.y * sin(theta);
		double Y = pt1.x * sin(theta) + pt1.y * cos(theta);

		double angle = atan2(Y,X);
		if(angle < 0)
		{
			angle += (2 * PI);
		}
		//
		return (180 * angle / PI);
	}

	// calculate angle between 3 points
	static double angleBetween(Point2f previous,Point2f center, Point2f current) {

	  double res = (180 * (atan2(current.x - center.x,current.y - center.y)-
							atan2(previous.x- center.x,previous.y- center.y)) / PI);
	  return res;
	}

	static vector<double> calculateAngles(vector<Point2f> v)
	{
		vector<double> res;
		if(v.size() < 3)
		{
			return res;
		}
		for(int i = 0;i < v.size();i++)
		{
			res.push_back(angleBetween(v[i], v[(i + 1) % v.size()], v[(i + 2) % v.size()]));
		}
		return res;
	}

	static vector<vector<int> > getAnglePoints()
	{
		string all[] = {
			"3 5 1 6",
			"2 7 4 8",
			"11 13 9 14",
			"10 15 12 16",
			"20 21 19 22",
			"24 25 23 26",
			"24 27 23 28",
			"24 23 29"};

		vector<vector<int> > v;
		for(int i = 0;i < 8;i++)
		{
			istringstream istr(all[i]);
			int k;
			vector<int> tmp;
			while(istr >> k)
			{
				tmp.push_back(k - 1);
			}
			v.push_back(tmp);
		}

		return v;
	}

	static void trainAngles(vector<vector<Fiducial> >& v,string name)
	{
		// get the angles
		vector<vector<int> > angles = Training::getAnglePoints();
		// positive and negative samples
		vector<vector<double> > pos,neg;
		for(int i = 0;i < v.size();i++)
		{
			vector<Fiducial> fids = v[i];
			vector<double> pos_sample,neg_sample;
			// display angles for first one
			
			for(int k = 0;k < angles.size();k++)
			{
				//vector<int> vv = angles[k];
				vector<Point2f> pts;
				for(int j = 0;j < angles[k].size();j++)
				{
					pts.push_back(Point2f(fids[angles[k][j]].x,fids[angles[k][j]].y));
				}
				// positive sample
				vector<double> tmp = Training::calculateAngles(pts);
				pos_sample.insert(pos_sample.end(),tmp.begin(), tmp.end());
			}
			// generate negative sample
			int ind = 0;
			for(int k = 0;k < angles.size();k++)
			{
				vector<Point2f> pts;
				for(int j = 0;j < angles[k].size();j++)
				{
					if(rand()&1)
					{
						pts.push_back(Point2f(fids[angles[k][j]].x + (rand() % 21) - 10,
							fids[angles[k][j]].y + (rand() % 21) - 10));
					}
					else
					{
						pts.push_back(Point2f(fids[angles[k][j]].x, fids[angles[k][j]].y));
					}
				}
				// negative sample
				vector<double> tmp = Training::calculateAngles(pts);
				neg_sample.insert(neg_sample.end(),tmp.begin(), tmp.end());
			}
			// add the samples
			pos.push_back(pos_sample);
			neg.push_back(neg_sample);
		}
		// start training
		// first choose parameters for training
		char ff[20];
		sprintf(ff,"%s",name.c_str());
		ofstream trainf(ff);
		for(int i = 0;i < pos.size();i++)
		{
			trainf << "+1";
			for(int j = 0;j < pos[i].size();j++)
			{
				trainf << " " << j + 1 << ":" << pos[i][j];
			}
			trainf << endl;
		}
		for(int i = 0;i < neg.size();i++)
		{
			trainf << "-1";
			for(int j = 0;j < neg[i].size();j++)
			{
				trainf << " " << j + 1 << ":" << neg[i][j];
			}
			trainf << endl;
		}
		trainf.close();
		// then start training
		train_with_cross_validation(ff);
	}
};
