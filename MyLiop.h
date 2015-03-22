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


#ifndef MYLIOP_H
#define MYLIOP_H

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
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <cstring>
#include <map>
#include <vector>
#include <sstream>
#include <cstring>
#include <vl/liop.h>
#include <vl/generic.h>
#include <vl/mathop.h>
#include <vl/imopv.h>


using namespace std;

Mat getLiopDescriptor(Mat &img, vector<KeyPoint>& keypoints)
{
	vl_size sideLength = 41 ;
	vl_size size = sideLength*sideLength ;
	float * patch = new float[size];
	// Create a new object instance (these numbers corresponds to parameter
	// values proposed by authors of the paper, except for 41)
	// side length must be odd
	VlLiopDesc * liop = vl_liopdesc_new_basic (sideLength);
	// allocate the descriptor array
	vl_size dimension = vl_liopdesc_get_dimension(liop) ;
	Mat desc = Mat::zeros(keypoints.size(), dimension, CV_32F);
	// convert image to gray
	Mat gray;
	cvtColor(img, gray, CV_RGB2GRAY);
	for(int i = 0;i < keypoints.size();i++)
	{
		KeyPoint k = keypoints[i];
		Mat rot_mat = getRotationMatrix2D(k.pt,k.angle,1);
		/// Rotate the gray image
	 	Mat dst;
		warpAffine( gray, dst, rot_mat, gray.size() );
		// extract patch
		int half = sideLength / 2;
		for(int r = 0, rp = k.pt.y - half;r < sideLength;r++,rp++)
		{
			for(int c = 0, cp = k.pt.x - half;c < sideLength;c++,cp++)
			{
				if(rp < 0 || rp >= dst.rows || cp < 0 || cp >= dst.cols)
				{
					patch[r * sideLength + c] = 0;
				}
				else
				{
					patch[r * sideLength + c] = ((unsigned char*)dst.data)[cp + rp * dst.cols];
				}
				//cout << r * sideLength + c << endl;
			}
		}
		// calculate descriptor
		vl_liopdesc_process(liop, ((float*)desc.data) + i * dimension, patch) ;
	}
	// delete the object
	vl_liopdesc_delete(liop) ;
	// delete the patch
	delete []patch;
	return desc;
}

void test()
{
  vl_int i ;
  vl_size sideLength = 21 ;
  vl_size size = sideLength*sideLength ;
  float mat[] = {
    6,6,6,6,6,6,6,6,6,6,6,
    6,6,6,5,4,4,4,5,6,6,6,
    6,6,5,4,3,3,3,4,5,6,6,
    6,5,4,3,2,2,2,3,4,5,6,
    6,4,3,2,2,1,2,2,3,4,6,
    6,4,3,2,1,1,1,2,3,4,6,
    6,4,3,2,2,1,2,2,3,4,6,
    6,5,4,3,2,2,2,3,4,5,6,
    6,6,5,4,3,3,3,4,5,6,6,
    6,6,6,5,4,4,4,5,6,6,6,
    6,6,6,6,6,6,6,6,6,6,6};
  float * patch = (float*)malloc(sizeof(float)*size);

  for(i = 0; i < (signed)size; i++){
    patch[i] = mat[i%121];
  }

  
	// Create a new object instance (these numbers corresponds to parameter
	// values proposed by authors of the paper, except for 41)
	//
	VlLiopDesc * liop = vl_liopdesc_new_basic (sideLength);
	// allocate the descriptor array
	vl_size dimension = vl_liopdesc_get_dimension(liop) ;
	float * desc = (float*)malloc(sizeof(float) * dimension) ;
	cout << dimension << endl;
	// compute descriptor from a patch (an array of length sideLegnth *
	// sideLength)
	vl_liopdesc_process(liop, desc, patch) ;
	// delete the object
	vl_liopdesc_delete(liop) ;
	free(patch);
}

#endif
