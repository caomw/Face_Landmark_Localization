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
#include "MyGa.h"
//#include <conio.h>
//#include <direct.h>

void MyPolygon()
{
	Mat img = imread("D://img1.jpg");
	int lineType = 8;
	int w = min(img.cols,img.rows);

	/** Create some points */
	Point rook_points[1][20];
	rook_points[0][0] = Point( w/4.0, 7*w/5.0 );
	rook_points[0][1] = Point( 3*w/4.0, 7*w/8.0 );
	rook_points[0][2] = Point( 3*w/4.0, 13*w/16.0 );
	rook_points[0][3] = Point( 11*w/16.0, 13*w/16.0 );
	rook_points[0][4] = Point( 19*w/20.0, 3*w/8.0 );
	rook_points[0][5] = Point( 3*w/4.0, 3*w/8.0 );
	rook_points[0][6] = Point( 3*w/4.0, w/8.0 );
	rook_points[0][7] = Point( 26*w/40.0, w/8.0 );
	rook_points[0][8] = Point( 26*w/40.0, w/4.0 );
	rook_points[0][9] = Point( 22*w/40.0, w/4.0 );
	rook_points[0][10] = Point( 22*w/40.0, w/8.0 );
	rook_points[0][11] = Point( 18*w/40.0, w/8.0 );
	rook_points[0][12] = Point( 18*w/40.0, w/4.0 );
	rook_points[0][13] = Point( 14*w/40.0, w/4.0 );
	rook_points[0][14] = Point( 14*w/40.0, w/8.0 );
	rook_points[0][15] = Point( w/4.0, w/8.0 );
	rook_points[0][16] = Point( w/4.0, 3*w/8.0 );
	rook_points[0][17] = Point( 13*w/32.0, 3*w/8.0 );
	rook_points[0][18] = Point( 5*w/16.0, 13*w/16.0 );
	rook_points[0][19] = Point( w/4.0, 13*w/16.0) ;

	const Point* ppt[1] = { rook_points[0] };
	int npt[] = { 20 };

	fillPoly( img,
			ppt,
			npt,
			1,
			Scalar( 255, 255, 255 ),
			lineType );
	vector<Point> v;
	for(int i = 0;i < 20;i++)
	{
		v.push_back(rook_points[0][i]);
	}
	Fiducial f;
	vector<Point> ind;
	convexHull(v,ind);
	cout << ind.size() << endl;
	// then draw the convex
	for(int i = 0;i < ind.size();i++)
	{
		line(img,ind[i],ind[(i + 1) % ind.size()],cv::Scalar(0,0,255),2);
	}
	// then generate random points and check if they are inside
	for(int i = 0;i < 300;i++)
	{
		Point p = Point(rand()%img.cols,rand()%img.rows);
		int val = pointPolygonTest(v,p,true);
		char number[10];
		sprintf(number,"%d", val);
		circle(img,p,4,cv::Scalar(255,255,255),-1);
		if(val > 0)
		{
			// inside, green
			putText(img, number, p,FONT_HERSHEY_COMPLEX_SMALL, 0.4, cvScalar(0,255,0), 1, CV_AA);
		}
		else if(val < 0)
		{
			putText(img, number, p,FONT_HERSHEY_COMPLEX_SMALL, 0.4, cvScalar(255,0,0), 1, CV_AA);
		}
	}
	imshow("test", img);
	cvWaitKey(0);
}

Mat overlay_mage(string dat_file,vector<int> not_included)
{
	cout << dat_file << endl;
	ifstream inp(dat_file);
	string img_file;
	getline(inp,img_file);

	// replace the characters
	std::replace(img_file.begin(), img_file.end(), '\\', '/');
	
	vector<Point2f> detect;
	vector<Point2f> orig;
	int num;
	inp >> num;
	Point2f minp(100000,100000), maxp(0,0);
	for(int i = 0;i < num;i++)
	{
		Point2f p_d, p_o;
		inp >> p_d.x >> p_d.y >> p_o.x >> p_o.y;
		detect.push_back(p_d);
		orig.push_back(p_o);
		// get min and max
		minp.x = min(minp.x, p_o.x);
		minp.y = min(minp.y, p_o.y);
		maxp.x = max(maxp.x, p_o.x);
		maxp.y = max(maxp.y, p_o.y);
	}
	// get bounding square
	double width = maxp.x - minp.x;
	double height = maxp.y - minp.y;
	double length = max(width, height) * 1.5;
	Point2f pc((maxp.x + minp.x) / 2, (maxp.y + minp.y) / 2);
	cv::Rect rect = cv::Rect(pc.x - length / 2,pc.y - length / 2, length, length);
	// display
	Mat img = imread(img_file);
	//imshow("img",img);
	//cv::waitKey(0);
//
	cout << "read img file: " <<img_file << endl;	
cout << "(" << img.cols << "," << img.rows << ")" << endl;
cout << "(" << rect.x << "," << rect.y << ") (" << rect.width << "," << rect.height << ")" << endl;
	if(rect.x < 0) rect.x = 0;
	if(rect.y < 0) rect.y = 0;
	if(rect.x + rect.width >= img.cols) rect.width = img.cols - rect.x - 1;
	if(rect.y + rect.height >= img.rows) rect.height = img.rows - rect.y - 1;
	// then crop and move all points
cout << "(" << rect.x << "," << rect.y << ") (" << rect.width << "," << rect.height << ")" << endl;
	img = img(rect);
	double ratio = 500.0 / img.cols;
	cv::resize(img,img,cv::Size(img.cols * ratio,img.rows * ratio));
	//img = Testing::overlayFidsImage(img, orig);
	// then divide the points into 3 categories
	double inter_occular = cv::norm(Point(orig[16].x,orig[16].y)-Point(orig[17].x,orig[17].y));
	vector<Point2f> vv[3];
	for(int i = 0;i < detect.size();i++)
	{
		int j = 0;
		for(;j < not_included.size();j++)
		{
			if(not_included[j] == i)
			{
				break;
			}
		}
		if(j < not_included.size())
			continue;
		double err = cv::norm(Point(orig[i].x,orig[i].y)-Point(detect[i].x,detect[i].y)) / inter_occular;
		Point2f p((detect[i].x - rect.x) * ratio, (detect[i].y - rect.y) * ratio);
		orig[i].x = (orig[i].x - rect.x) * ratio;
		orig[i].y = (orig[i].y - rect.y) * ratio;
		if(err < 0.05)
		{
			// error less than 5 %
			vv[0].push_back(p);
		}
		else if(err < 0.1)
		{
			// error less than 10 %
			vv[1].push_back(p);
		}
		else
		{
			// other errors
			vv[2].push_back(p);
		}
	}
	img = Visualization::overlayFidsImage(img, orig,4);
	img = Visualization::overlayFidsImage(img,orig,cv::Scalar(255,255,255),true);
	//img = Testing::overlayFidsImage(img, vv[0],4);
	//img = Testing::overlayFidsImage(img, vv[1],4);
	//img = Testing::overlayFidsImage(img, vv[2],4);

	return img;
}

Mat displayImage(string name,vector<int> not_included,bool save = false)
{
	Mat img = overlay_mage(name + ".dat",not_included);
	if(save)
	{
		imwrite(name + ".jpg", img);
	}
	imshow("test", img);
	return img;
}

void display(string base)
{
	int choice;
	string fileName = "";
	int ind;
	while(!fileName.size())
	{
		puts("1 - Display LFPW");
		puts("2 - Display Helen");
		puts("3 - Display BioID");
		puts("4 - Hard old Belhumuer");
		cin >> choice;
		switch(choice)
		{
		case 1:
			{
				VideoWriter writer;
				writer.open("LFPW.avi", CV_FOURCC('D', 'I', 'V', 'X'),1,cv::Size(300,300));
				// 1 - 168
				puts("Choose image [1, 168]");
				cin >> ind;
				while(ind <= 168)
				{
					if(ind < 1)
					{
						ind = 168;
					}
					else if(ind > 168)
					{
						ind = 1;
					}
					cout << ind << endl;
					ostringstream os;
					os << base << "output/fid" << setw(4) << setfill('0') <<  ind;
					fileName = os.str();
cout << fileName << endl;
					Mat img = displayImage(fileName, vector<int>(),true);
					writer << img;
					int key = cv::waitKey(1000);
					if(key == 27) // esc
					{
						break;
					}
					else if(key == 2424832) // left
					{
						ind--;
					}
					else if(key == 2555904)
					{
						ind++;
					}
					else
					{
						ind++;
					}
				}
				destroyWindow("test");
				//cvReleaseVideoWriter( &writer );
			}
			break;
		case 4:
			{
				// 1 - 168
				puts("Choose image [1, 168]");
				cin >> ind;
				while(true)
				{
					if(ind < 1)
					{
						ind = 168;
					}
					else if(ind > 168)
					{
						ind = 1;
					}
					ostringstream os;
					os << base << "output_LFPW_hard/fid" << setw(4) << setfill('0') <<  ind;
					fileName = os.str();
					displayImage(fileName, vector<int>(),true);
					int key = cv::waitKey(0);
					if(key == 27) // esc
					{
						break;
					}
					else if(key == 2424832) // left
					{
						ind--;
					}
					else if(key == 2555904)
					{
						ind++;
					}
					else
					{
						ind++;
					}
				}
				destroyWindow("test");
			}
			break;
		case 2:
			{
				VideoWriter writer;
				writer.open("Helen.avi", CV_FOURCC('D', 'I', 'V', 'X'),1,cv::Size(300,300));
				// 1 - 348
				puts("Choose image [1, 348]");
				cin >> ind;
				while(ind <= 348)
				{
					if(ind < 1)
					{
						ind = 348;
					}
					else if(ind > 348)
					{
						ind = 1;
					}
					cout << ind << endl;
					ostringstream os;
					os << base << "helen_output//fid" << setw(4) << setfill('0') <<  ind;
					fileName = os.str();
					Mat img = displayImage(fileName,vector<int>(),true);
					writer << img;
					int key = cv::waitKey(1000);
					if(key == 27) // esc
					{
						break;
					}
					else if(key == 2424832) // left
					{
						ind--;
					}
					else if(key == 2555904)
					{
						ind++;
					}
					else
					{
						ind++;
					}
				}
				destroyWindow("test");
				//cvReleaseVideoWriter( &writer );
			}
			break;
		case 3:
			{
				VideoWriter writer;
				writer.open("BioID.avi", CV_FOURCC('D', 'I', 'V', 'X'),1,cv::Size(300,300));
				// 1 - 1520
				puts("Choose image [1, 1520]");
				cin >> ind;
				while(true)
				{
					if(ind < 1)
					{
						ind = 1520;
					}
					else if(ind > 1520)
					{
						ind = 1;
					}
					cout << ind << endl;
					ostringstream os;
					os << base << "bioid_output//fid" << setw(4) << setfill('0') <<  ind;
					fileName = os.str();
					static int map_ind[] = {1,2,3,4,0,0,0,0,9,10,11,12,0,0,0,0,17,18,19,20,21,0,23,24,25,0,0,28,0};
					vector<int> _not;
					for(int i = 0;i < 29;i++)
					{
						if(!map_ind[i])
							_not.push_back(i);
					}
					Mat img = displayImage(fileName,_not,true);
					writer << img;
					int key = cv::waitKey(0);
					if(key == 27) // esc
					{
						break;
					}
					if(key == 2424832) // left
					{
						ind--;
					}
					if(key == 2555904)
					{
						ind++;
					}
				}
				destroyWindow("test");
//				cvReleaseVideoWriter( &writer );
			}
			break;
		default:
			puts("Incorrect choice!");
			break;
		}
	}
}

int vid( ) 
{
	/*Here i do not use arguments to main, if you like you may use the same....*/
 
	double fps=0.5;/*As is the general fps rate.*/
 
	/*CvSize is a basic structure which has 2 coordinates which , as the name
	indicates can store the size which is nothing but the width
	and the height of the frame and is generally used for the same.*/
 
	CvSize size = cvSize(320,240);
 
	IplImage *image=cvLoadImage("filename one.JPG",1); /*here please substitute the
	 name of your file which you have with you
	with the extension...do the same whenever you see cvLoadImage();*/
 
	if(image==NULL ) {
 
		puts("unable to load the frame");
		//getch();
		exit(0);
	}
 
	/*This next line is very important it will initialize a videowriter as the name indicates*/
 
	/*first argument is the name of the video.The string can also include the full path with the string name
 
	Ex:"e:/video.avi",second argument:4-character code of codec used to compress the frames.
 
	fps:
 
	Framerate of the created video stream.
 
	frame_size
 
	Size of video frames. Instead of "'D','I','V','X'" YOU CAN TRY to use some
 
	other codecs as told in the documentation
	But this codec is working for me so i use this one...*/
 
	CvVideoWriter* writer = cvCreateVideoWriter("Video from Images.avi",CV_FOURCC('D','I','V','X'),fps,size);
 
	//------------------------------------------------------------
 
	/*Till the count reaches 300 keep on writing frames..
 
	if you do not want to think much then you may change the upper limit of the
	 counter and see the result
	try to find out the relation of the counter with the "fps"*/
 
	for(int counter=0;counter < 300;counter++)
 
	{
 
		printf("Enteredn");/*Statements like these are very helpful in
		knowing which loops has the program entered and are helpful in debugging the program.*/
 
		/*The below statement writes the frame one by one to the video ...*/
		cvWriteFrame( writer, image);
	}
	image=cvLoadImage("filename2.JPG",1);
	if(image==NULL ) 
	{
		/*if you are not able to understand these lines
		please study some of my previous posts*/
 
		puts("unable to load the second frame");
		cvWaitKey(0);
		exit(0);
	}
 
	for(int counter=0;counter < 300;counter++)/*here the counter in both the
	images is same..which means that the
	images will show up in the video in equal times...*/
	{
	printf("Entered-2nd-loop");
 
	cvWriteFrame( writer,image);
	} cvWaitKey(0);/*wait till a key is pressed..*/
	cvReleaseVideoWriter( &writer );/*releasing the Video writer
	which is the same as we release the image..or any other structure*/
 
	cvReleaseImage( &image );
}
 

int main(int argc, char*argv[])
{
	/*Training::trainAllFidPoints("lfw//","train");
	return 0;
	double pstart = atof(argv[1]);
	double pend = atof(argv[2]);
	double pstep = atof(argv[3]);

	for(double mut = pstart;mut <= pend;mut+=pstep)
	{
		string str = "CalcErrors.exe ";
		str += GA::getInst()->testLFPW("","lfw//", "test",1,168,0.9,0.1,30,400);
		system(str.c_str());
	}*/
	int start,end;
	int choice;
	bool run = false;
	string base = "/home/mostafaizz/";///media/mostafaizz/New Volume/mywork/Masters/person_reidentification/Gradient_proj/x64";
	while(!run)
	{
		puts("1 - LFPW");
		puts("2 - Helen");
		puts("3 - BioID");
		puts("4 - OLD Belhumuer on LFPW");
		puts("5 - Display");
		cin >> choice;
		switch(choice)
		{
		case 1:
			run = true;
			// 1 - 168
			puts("Choose start and end [1, 168]");
			cin >> start >> end;
			GA::getInst()->testLFPW(base,"lfw/", "test",start,end);
			break;
		case 2:
			run = true;
			// 1 - 348
			puts("Choose start and end [1, 348]");
			cin >> start >> end;
			GA::getInst()->testHelen(base,"lfw/", "Helen/",start,end);
			break;
		case 3:
			run = true;
			// 1 - 1520
			puts("Choose start and end [1, 1520]");
			cin >> start >> end;
			GA::getInst()->testBioID(base,"lfw/","BioID-FaceDatabase/",start,end);
			break;
		case 5:
			// display
			display(base);
			break;
		case 4:
			Testing::testFids("lfw/", "test",1);
			break;
		default:
			puts("Incorrect choice!");
			break;
		}
	}
	//Testing::testFidsHillClimbing("D://Personal//Work//Masters//lfw//", "test",1);
	//sortImages("D://Personal//Work//Masters//lfw//", "train");
	// load all fiducial points for the training
	//vector<vector<Fiducial> > all_fids = sortFidPointsByPose(loadAllFidPoints("D://Personal//Work//Masters//lfw//", "train"));
	//vector<svm_model* > svm_models = loadSVMModels();
	//testFidsOneImage(svm_models, all_fids, "D:/img1.jpg");
	//testBioID("D://Personal//Work//Masters//BioID//BioID-FaceDatabase//");

	//vector<FaceData> allFaces = Loading::loadAllFaces("D://Personal//Work//Masters//lfw//", "test");
	
	return 0;
}
