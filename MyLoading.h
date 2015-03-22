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


#ifndef MY_LOADING
#define MY_LOADING

#include <vector>
#include <iomanip>
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
#include "mylibsvm.h"

using namespace std;
using namespace cv;

enum Visibility
{
    Visible = 0,
    obscured_by_hair_glasses_etc = 1,
    hidden_because_of_viewing_angle = 2,
    hidden_because_of_image_crop = 3
};

struct Fiducial
{
	double x,y;
	double var;
	Visibility t;
	friend ostream& operator<<(ostream& ,const Fiducial&);
};

ostream& operator<<(ostream& ost,const Fiducial& fid)
{
	ost << "(" << fid.x << "," << fid.y << ") = " << fid.t;
	return ost;
}

struct DetecetedPoint
{
	double x,y;
	double prob; // the probability we got from the SVM
};

// find the rectangle around face
void getFaceRectangle(cv::Mat &img,int& minx,int& miny,int maxx,int maxy, int &dx, int &dy)
{
	// x
	minx = max(0, minx - 20);
	maxx = min(img.cols - 1, maxx + 20);
	dx = maxx - minx + 1;
	// y
	miny = max(0, miny - 20);
	maxy = min(maxy + 20, img.rows - 1);
	dy = maxy - miny + 1;
}

struct FaceData
{
	vector<Fiducial> fids_orig;
	Rect rect;
	string path;
	string ID;
	double factor;
	bool crop;
	int minx, miny, maxx,maxy;
	bool mirror;
	FaceData()
	{
		mirror = false;
		minx = 100000,maxx = 0,miny = 100000,maxy = 0;
	}
	// load the image
	Mat get_img()
	{
		Mat img = imread(path);
		if(mirror)
		{
			cv::flip(img,img,1);
		}
		crop_and_scale(img);
		return img;
	}
	// load the original image
	Mat get_orig_img()
	{
		Mat img = imread(path);
		if(mirror)
		{
			cv::flip(img,img,1);
		}
		return img;
	}
	void make_mirror()
	{
		mirror = ! mirror;
		Mat img = imread(path);
		// flip the fiducial points
		int mirror[] = {1,0,3,2,6,7,4,5,9,8,11,10,14,15,12,13,17,16,19,18,20,21,23,22,24,25,26,27,28};
		vector<Fiducial> fids(29);
		for(int j = 0;j < fids.size();j++)
		{
			fids[j] = (this->fids_orig[mirror[j]]);
			fids[j].x = img.cols - fids[j].x - 1;
		}
		this->fids_orig = fids;
		int tmp = img.cols - maxx - 1;
		maxx = img.cols - minx - 1;
		minx = tmp;
	}

	void crop_and_scale(Mat &img)
	{
		//orig = img.clone();
		// scale to inter_ocular = 55
		double interOccular = abs(fids_orig[16].x - fids_orig[17].x);
		factor = 55.0 / interOccular;
		cv::resize(img,img,cvSize(img.cols*factor,img.rows*factor));
		// find the face rectangle
		//rect.width = img.cols;
		//rect.height = img.rows;
		rect.x = minx * factor;
		rect.y = miny * factor;
		getFaceRectangle(img,rect.x,rect.y,maxx*factor,maxy*factor,rect.width,rect.height);
		// get the face rectangle
		if(crop)
		{
			img = img(rect);
		}
	}

	vector<Fiducial> get_scaled_fids()
	{
		Mat img = imread(path);
		vector<Fiducial> fids = this->fids_orig;
		double interOccular = abs(fids[16].x - fids[17].x);
		factor = 55.0 / interOccular;
		cv::resize(img,img,cvSize(img.cols*factor,img.rows*factor));
		// find the face rectangle
		//rect.width = img.cols;
		//rect.height = img.rows;
		rect.x = minx * factor;
		rect.y = miny * factor;
		getFaceRectangle(img,rect.x,rect.y,maxx*factor,maxy*factor,rect.width,rect.height);
		// move the fid points as well to (minx, miny) and resize them with the same inter_ocular ratio
		for(int i = 0;i < fids.size();i++)
		{
			//Fiducial p = fids[i];
			if(crop)
			{
				fids[i].x = (fids[i].x * factor) - rect.x;
				fids[i].y = (fids[i].y * factor) - rect.y;
			}
			else
			{
				fids[i].x = (fids[i].x * factor);
				fids[i].y = (fids[i].y * factor);
			}
		}
		return fids;
	}

	template<class T>
	vector<T> get_original(vector<T> inp,string fileName = "")
	{
		vector<T> res = inp;
		if(crop)
		{
			for(int i = 0;i < inp.size();i++)
			{
				res[i].x = (res[i].x + rect.x) / factor;
				res[i].y = (res[i].y + rect.y) / factor;
			}
		}
		else
		{
			for(int i = 0;i < inp.size();i++)
			{
				res[i].x = res[i].x / factor;
				res[i].y = res[i].y / factor;
			}
		}
		// write to file if file name exists
		if(fileName.size() > 0)
		{
			ofstream oup(fileName);
			oup << path << endl;
			oup << res.size() << endl;
			for(int i = 0;i < res.size();i++)
			{
				// x_detected, y_detected, x_orig, y_orig
				oup << res[i].x << " " << res[i].y << " " << fids_orig[i].x << " " << fids_orig[i].y << endl;
			}
			oup.close();
		}
		return res;
	}
};

class Loading
{
public:
	// load one svm model with index i
	static svm_model* loadSVMModelNumber(int i,string name)
	{
		// load the model
		char ff[100];
		sprintf(ff,"%s%.2d.model",name.c_str(), i);
		return svm_load_model(ff);
	}

	// load geenral svm model with
	static svm_model* loadSVMModel(string name)
	{
		// load the model
		char ff[100];
		sprintf(ff,"%s.model",name.c_str());
		return svm_load_model(ff);
	}

	// load all svm models
	static vector<svm_model*> loadSVMModels(string name = "fid")
	{
		cout << "loading SVM models" << endl;
		vector<svm_model*> models;
		for(int i = 0;i < 29;i++)
		{
			models.push_back(loadSVMModelNumber(i,name));
		}
		// 
		cout << "end loading models" << endl;
		return models;
	}
	
	// load fiducial point from ceratin file
	static vector<Fiducial> loadFiducial(string labels, int &minx, int &miny, int &maxx, int &maxy)
	{
		ifstream flab(labels.c_str());
		string tmp;
		flab >> tmp >> tmp;
		double x,y;
		int ind;
		int cnt = 0;
		vector<Fiducial> fids;
		while(flab >> x >> y >> ind)
		{
			if(fids.size() == 29)
			{
				fids.erase(fids.begin() + 28,fids.end());
			}
			Fiducial fid;
			fid.x = x; fid.y = y; fid.t = (Visibility)ind;
			fids.push_back(fid);
		}
		for(int i = 0;i < fids.size();i++)
		{
			minx = min(minx,(int)fids[i].x);
			miny = min(miny,(int)fids[i].y);
			maxx = max(maxx,(int)fids[i].x);
			maxy = max(maxy,(int)fids[i].y);
		}
		return fids;
	}

	// load fiducial point from ceratin file
	static vector<Fiducial> loadFiducialBioID(string labels, int &minx, int &miny, int &maxx, int &maxy)
	{
		ifstream flab(labels.c_str());
		vector<Fiducial> fids;
		string tmp;
		bool start = false;
		while(getline(flab,tmp))
		{
			if(start)
			{
				if(tmp == "}")
				{
					// end 
					break;
				}
				Fiducial fid;
				istringstream inp(tmp);
				char c;
				inp >> fid.x >> fid.y;
				fids.push_back(fid);
			}
			if(tmp == "{")
			{
				start = true;
			}			
		}
		for(int i = 0;i < fids.size();i++)
		{
			minx = min(minx,(int)fids[i].x);
			miny = min(miny,(int)fids[i].y);
			maxx = max(maxx,(int)fids[i].x);
			maxy = max(maxy,(int)fids[i].y);
		}
		return fids;
	}


	// load fiducial point from ceratin file
	static vector<Fiducial> loadFiducialHelen(string labels, int &minx, int &miny, int &maxx, int &maxy)
	{
		ifstream flab(labels.c_str());
		vector<Fiducial> fids;
		string tmp;
		while(getline(flab,tmp))
		{
			Fiducial fid;
			istringstream inp(tmp);
			char c;
			inp >> fid.x>> c >> fid.y;
			fids.push_back(fid);
		}
		for(int i = 0;i < fids.size();i++)
		{
			minx = min(minx,(int)fids[i].x);
			miny = min(miny,(int)fids[i].y);
			maxx = max(maxx,(int)fids[i].x);
			maxy = max(maxy,(int)fids[i].y);
		}
		return fids;
	}

	// vector<vector<Fiducial> >
	static vector<vector<Fiducial> > sortFidPointsByPose(vector<vector<Fiducial> > allfids)
	{
		// map for sorted images
		multimap<double,vector<Fiducial> > faceMap;
	
		for(int i = 0;i < allfids.size();i++)
		{	
			vector<Fiducial> fids = allfids[i];

			double d1 = (fids[8].x - fids[10].x) * (fids[8].x - fids[10].x) + (fids[8].y - fids[10].y) * (fids[8].y - fids[10].y);
			double d2 = (fids[9].x - fids[11].x) * (fids[9].x - fids[11].x) + (fids[9].y - fids[11].y) * (fids[9].y - fids[11].y);
			faceMap.insert(make_pair(d1 / d2, fids));
		}
	
		vector<vector<Fiducial> > res;
		for(multimap<double,vector<Fiducial> >::iterator itr = faceMap.begin();itr != faceMap.end();itr++)
		{
			res.push_back(itr->second);
		}

		return res;
	}

	// get the pose ratio to be used in sorting
	static double getPoseRatio(vector<Fiducial> &fids)
	{
		double d1 = (fids[19].x - fids[20].x) * (fids[19].x - fids[20].x) 
			+ (fids[19].y - fids[20].y) * (fids[19].y - fids[20].y);
		double d2 = (fids[18].x - fids[20].x) * (fids[18].x - fids[20].x) 
			+ (fids[18].y - fids[20].y) * (fids[18].y - fids[20].y);
		
		return d1/d2;
	}

	static vector<FaceData> sortFacesByPose(vector<FaceData> & allfids)
	{
		// map for sorted images
		multimap<double,FaceData > faceMap;
	
		for(int i = 0;i < allfids.size();i++)
		{	
			double ratio = getPoseRatio(allfids[i].fids_orig);
			faceMap.insert(make_pair(ratio, allfids[i]));
		}
	
		vector<FaceData> res;
		for(multimap<double,FaceData >::iterator itr = faceMap.begin();itr != faceMap.end();itr++)
		{
			res.push_back(itr->second);
		}

		return res;
	}

	// load all fiducial points for all examples available
	static vector<vector<Fiducial> > loadAllFidPoints(string Folder, string type,bool calcCov = true)
	{
		vector<vector<Fiducial> > res;

		ostringstream str;
		// reading the file describing all images
		str << Folder << type << "/" << type << ".txt";
		string file = str.str();
		ifstream inp(file.c_str());
		string line;
		// find the correct files
		//ofstream correct(Folder + type + "/correct.txt");
		// read correct files only
		try
		{
			while(getline(inp,line,'\n'))
			{
				//cout << inp << endl;
				istringstream lineInp(line);
				vector<string> v; // all images
				string tmp;
				while(lineInp >> tmp)
				{
					v.push_back(tmp);
				}
				// display images only now
				string imgName = "";
				int minx = 10000,miny = 10000,maxx = 0,maxy = 0;
		
				// load the labels
				string labels = Folder + type + "/" + v[0] + "/average.txt";
				// load fiducial points from file
				vector<Fiducial> fids = loadFiducial(labels, minx, miny, maxx, maxy);
		
				vector<Fiducial> fids_mir = fids;
				// get the Mirror Image
				double dc = 0;
				for(int i = 0;i < fids.size();i++)
				{
					dc +=fids[i].x;
				}
				dc /= fids.size();
				for(int i = 0;i < fids.size();i++)
				{
					fids_mir[i].x -= dc;
					fids_mir[i].x = -fids_mir[i].x;
					fids_mir[i].x += dc;
				}
				res.push_back(fids);
				// and add the mirror
				//res.push_back(fids_mir);
			}
			cout << "All Fid points loaded : " << res.size() << endl;
			// getting the variance for each point
			if(calcCov)
			{
				cout << "calculating covariance" << endl;
				for(int i = 0; i < res.size();i++)
				{
					double inter_occular = sqrt((res[i][16].x - res[i][17].x) * (res[i][16].x - res[i][17].x) +
						(res[i][16].y - res[i][17].y) * (res[i][16].y - res[i][17].y));
					double bestError = 10000000000000000L;
					vector<double> var;
					for(int j = 0;j < 10;j++)
					{
						int r;
						do{ // get other sample
							r = rand() % res.size();
						}while(r == i);
						// get the best transformation
						// find best homography
						vector<Point2f> srcPts(res[i].size());
						vector<Point2f> dstPts(res[i].size());

						for(int k = 0 ;k < res[i].size();k++)
						{
							/// Set your 3 points to calculate the  Affine Transform
							dstPts[k] = Point2f( res[i][k].x,res[i][k].y);
							// set  destination point
							srcPts[k] = Point2f(res[r][k].x, res[r][k].y);
						}

						Mat warp_mat = findHomography( srcPts, dstPts );
						warp_mat.convertTo(warp_mat,CV_32FC1,1,0); //NOW A IS FLOAT 
						vector<Point2f> dst(res[i].size());
						// transform the points
						perspectiveTransform(srcPts,dst,warp_mat);
						// calculate the error
						vector<double> testVar;
						double error = 1;
						for(int k = 0;k < res[i].size();k++)
						{
							double distance = (srcPts[k].x - dstPts[k].x) * (srcPts[k].x - dstPts[k].x) + 
								(srcPts[k].y - dstPts[k].y) * (srcPts[k].y - dstPts[k].y);
							testVar.push_back(sqrt(distance) / inter_occular);
							error += distance;
						}
						error /= res[i].size();
						if(error < bestError)
						{
							bestError = error;
							var = testVar;
						}
					}
					// then update the fiducial points
					for(int k = 0;k < res[i].size();k++)
					{
						res[i][k].var = var[k];
					}
				}
				// end of variance calculations
				cout << "end of covariance calculations" << endl;
			}
		}
		catch(...)
		{
			// loading all fiducial points for all samples failed
			cout << "loading all fiducial points for all samples failed" << endl;
		}
	
		return res;
	}

	static vector<vector<Fiducial> > loadAllFidPointsWithMirrors(string Folder, string type,bool calcCov = true)
	{
		vector<vector<Fiducial> > res = loadAllFidPoints(Folder,type,calcCov);
		for(int i = res.size() - 1; i >= 0;i--)
		{
			// flip the fiducial points
			int mirror[] = {1,0,3,2,6,7,4,5,9,8,11,10,14,15,12,13,17,16,19,18,20,21,23,22,24,25,26,27,28};
			vector<Fiducial> fids(29);
			double xc = 0;
			for(int j = 0;j < res[i].size();j++)
			{
				xc += res[i][j].x;
			}
			xc /= res[i].size();
			for(int j = 0;j < res[i].size();j++)
			{
				fids[j] = res[i][mirror[j]];
				fids[j].x = 2 * xc - fids[j].x;
			}
			res.push_back(fids);
		}
		cout << "Now fids size = " << res.size() << endl;
		return res;
	}

	static void sortFids(vector<vector<Fiducial> > &all_fids)
	{
		// map for sorted fids
		multimap<double,vector<Fiducial> > fidsMap;
	
		for(int i = 0;i < all_fids.size();i++)
		{	
			double ratio = getPoseRatio(all_fids[i]);
			fidsMap.insert(make_pair(ratio, all_fids[i]));
		}
		int ind = 0;
		for(multimap<double,vector<Fiducial> >::iterator itr = fidsMap.begin();itr != fidsMap.end();itr++)
		{
			all_fids[ind++] = (itr->second);
		}
	}

	// load range for scaling
	static void loadRange(double &miny, double &maxy,vector<double>& minv,vector<double>& maxv, int fidNum,string name = "fid")
	{
		char range_f[100];
		sprintf(range_f,"%s%.2d.range", name.c_str(),fidNum);
		ifstream inp(range_f);
		char tmpc;
		int tmpind;
		double tempMinV, tempMaxV;
		inp >> tmpc;
		inp >> miny >> maxy;
		while(inp >> tmpind >> tempMinV >> tempMaxV)
		{
			minv.push_back(tempMinV);
			maxv.push_back(tempMaxV);
		}
		inp.close();
	}

	// load general range for scaling
	static void loadGeneralRange(double &miny, double &maxy,vector<double>& minv,vector<double>& maxv,string name)
	{
		char range_f[100];
		sprintf(range_f,"%s.range", name.c_str());
		ifstream inp(range_f);
		char tmpc;
		int tmpind;
		double tempMinV, tempMaxV;
		inp >> tmpc;
		inp >> miny >> maxy;
		while(inp >> tmpind >> tempMinV >> tempMaxV)
		{
			minv.push_back(tempMinV);
			maxv.push_back(tempMaxV);
		}
		inp.close();
	}

	// load all BioID data
	// D:\Personal\Work\Masters\BioID\BioID-FaceDatabase\

	static vector<FaceData> loadAllHelenFaces(string location,bool crop = true)
	{
		vector<FaceData> allFaces;
		try
		{
			printf("Loading...");
			int ind = 1;
			while(true)
			{
				// display images only now
				ostringstream imgName;
				// all images are in png format
				imgName << location << ind << ".png";
				ostringstream labels;
				// csv file
				labels << location << ind << ".csv";
				//
				ind++;
				//int minx = 1000000, miny = 1000000, maxx = 0,maxy = 0;
		
				//cout << "Loading img " << imgName << endl;
				printf(".");
				FaceData fd;
				fd.crop = crop;
				fd.path = imgName.str();
				Mat img = cv::imread(fd.path);
				if(img.rows == 0)
				{
					break;
				}
				// load the labels
				// load fiducial points from file
				fd.fids_orig = loadFiducialHelen(labels.str(), fd.minx, fd.miny, fd.maxx,fd.maxy);
				//fd.crop_and_scale();
				allFaces.push_back(fd);
			}
		}
		catch(...)
		{
			puts("Done!");
		}
		puts("Done!!!");
		return allFaces;
	}

	// load helen data set
	static vector<FaceData> loadAllBioIDFaces(string location,bool crop = true)
	{
		vector<FaceData> allFaces;
		try
		{
			printf("Loading...");
			int ind = 1;
			while(true)
			{
				// display images only now
				ostringstream imgName;
				// all images are in png format
				imgName << location << "BioID_" << setw(4) << setfill('0') <<  ind << ".pgm";
				ostringstream labels;
				// data file
				labels << location << "BioID_" << setw(4) << setfill('0') <<  ind << ".pts";
				//
				ind++;
				//int minx = 1000000, miny = 1000000, maxx = 0,maxy = 0;
		
				//cout << "Loading img " << imgName << endl;
				printf(".");
				FaceData fd;
				fd.path = imgName.str();
				fd.crop = crop;
				Mat img = cv::imread(fd.path);
				if(img.rows == 0)
				{
					break;
				}
				// load the labels
				// load fiducial points from file
				vector<Fiducial> temp_fids = loadFiducialBioID(labels.str(), fd.minx, fd.miny, fd.maxx, fd.maxy);
				static int map_ind[20] = {17,18,23,24,1,3,4,2,0,9,11,12,10,0,21,19,20,25,28,0};
				fd.fids_orig.resize(29);
				for(int i = 0;i < temp_fids.size();i++)
				{
					if(map_ind[i])
					{
						fd.fids_orig[map_ind[i] - 1] = temp_fids[i];
					}
				}
				//fd.crop_and_scale();
				allFaces.push_back(fd);
			}
		}
		catch(...)
		{
			puts("Done!");
		}
		puts("Done!!!");
		return allFaces;
	}

	static vector<FaceData> loadAllFaces(string Folder, string type,bool crop = true)
	{
		vector<FaceData> allFaces;
		ostringstream str;
		// reading the file describing all images
		str << Folder << type << "/" << type << ".txt";
		string file = str.str();
		ifstream inp(file.c_str());
		string line;
		// find the correct files
		//ofstream correct(Folder + type + "/correct.txt");
		// read correct files only
		string tmpStr = Folder + type + "/correct.txt";
		ifstream correct(tmpStr.c_str());

		printf("Loading...");
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
			FaceData fd;
			string imgName = "";
			fd.path = Folder + type + "/" + v[0] + "/img." + v[2];
			//cout << "Loading img " << imgName << endl;
			printf(".");
			fd.crop = crop;
			Mat img = cv::imread(fd.path);
			// load the labels
			string labels = Folder + type + "/" + v[0] + "/average.txt";
			// load fiducial points from file
			fd.fids_orig = loadFiducial(labels, fd.minx, fd.miny, fd.maxx, fd.maxy);
			//fd.crop_and_scale();
			allFaces.push_back(fd);
		}
		correct.close();
		puts("Done!");
		return allFaces;
	}

	static vector<FaceData> loadAllFacesSorted(string Folder, string type)
	{
		// load all files
		vector<FaceData> allFaces = loadAllFaces(Folder, type);
		// sort 
		return sortFacesByPose(allFaces);
	}

	static vector<FaceData> loadAllFacesWithMirrorsSorted(string Folder, string type,bool crop)
	{
		// load all files
		vector<FaceData> allFaces = loadAllFaces(Folder, type, crop);
		for(int i = allFaces.size() - 1;i >= 0;i--)
		{
			FaceData fd = allFaces[i];
			fd.make_mirror();
			allFaces.push_back(fd);
		}
		// sort 
		return sortFacesByPose(allFaces);
	}
};

#endif
