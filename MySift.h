#ifndef MYSIFT_H
#define MYSIFT_H
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/**********************************************************************************************\
 Implementation of SIFT is based on the code from http://blogs.oregonstate.edu/hess/code/sift/
 Below is the original copyright.

//    Copyright (c) 2006-2010, Rob Hess <hess@eecs.oregonstate.edu>
//    All rights reserved.

//    The following patent has been issued for methods embodied in this
//    software: "Method and apparatus for identifying scale invariant features
//    in an image and use of same for locating an object in an image," David
//    G. Lowe, US Patent 6,711,293 (March 23, 2004). Provisional application
//    filed March 8, 1999. Asignee: The University of British Columbia. For
//    further details, contact David Lowe (lowe@cs.ubc.ca) or the
//    University-Industry Liaison Office of the University of British
//    Columbia.

//    Note that restrictions imposed by this patent (and possibly others)
//    exist independently of and may be in conflict with the freedoms granted
//    in this license, which refers to copyright of the program, not patents
//    for any methods that it implements.  Both copyright and patent law must
//    be obeyed to legally use and redistribute this program and it is not the
//    purpose of this license to induce you to infringe any patents or other
//    property right claims or to contest validity of any such claims.  If you
//    redistribute or use the program, then this license merely protects you
//    from committing copyright infringement.  It does not protect you from
//    committing patent infringement.  So, before you do anything with this
//    program, make sure that you have permission to do so not merely in terms
//    of copyright, but also in terms of patent law.

//    Please note that this license is not to be understood as a guarantee
//    either.  If you use the program according to this license, but in
//    conflict with patent law, it does not mean that the licensor will refund
//    you for any losses that you incur if you are sued for your patent
//    infringement.

//    Redistribution and use in source and binary forms, with or without
//    modification, are permitted provided that the following conditions are
//    met:
//        * Redistributions of source code must retain the above copyright and
//          patent notices, this list of conditions and the following
//          disclaimer.
//        * Redistributions in binary form must reproduce the above copyright
//          notice, this list of conditions and the following disclaimer in
//          the documentation and/or other materials provided with the
//          distribution.
//        * Neither the name of Oregon State University nor the names of its
//          contributors may be used to endorse or promote products derived
//          from this software without specific prior written permission.

//    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
//    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
//    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
//    HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
\**********************************************************************************************/

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
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching/stitcher.hpp>

using namespace cv;
using namespace std;

/******************************* Defs and macros *****************************/

	// default number of sampled intervals per octave
	const int SIFT_INTVLS = 3;

	// default sigma for initial gaussian smoothing
	const float SIFT_SIGMA = 1.6f;

	// default threshold on keypoint contrast |D(x)|
	const float SIFT_CONTR_THR = 0.04f;

	// default threshold on keypoint ratio of principle curvatures
	const float SIFT_CURV_THR = 10.f;

	// double image size before pyramid construction?
	const bool SIFT_IMG_DBL = true;

	// default width of descriptor histogram array
	const int SIFT_DESCR_WIDTH = 4;

	// default number of bins per histogram in descriptor array
	const int SIFT_DESCR_HIST_BINS = 8;

	// assumed gaussian blur for input image
	const float SIFT_INIT_SIGMA = 0.5f;

	// width of border in which to ignore keypoints
	const int SIFT_IMG_BORDER = 5;

	// maximum steps of keypoint interpolation before failure
	const int SIFT_MAX_INTERP_STEPS = 5;

	// default number of bins in histogram for orientation assignment
	const int SIFT_ORI_HIST_BINS = 36;

	// determines gaussian sigma for orientation assignment
	const float SIFT_ORI_SIG_FCTR = 1.5f;

	// determines the radius of the region used in orientation assignment
	const float SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR;

	// orientation magnitude relative to max that results in new feature
	const float SIFT_ORI_PEAK_RATIO = 0.8f;

	// determines the size of a single descriptor orientation histogram
	const float SIFT_DESCR_SCL_FCTR = 3.f;

	// threshold on magnitude of elements of descriptor vector
	const float SIFT_DESCR_MAG_THR = 0.2f;

	// factor used to convert floating-point descriptor to unsigned char
	const float SIFT_INT_DESCR_FCTR = 512.f;

	#if 0
	// intermediate type used for DoG pyramids
	typedef short sift_wt;
	const int SIFT_FIXPT_SCALE = 48;
	#else
	// intermediate type used for DoG pyramids
	typedef float sift_wt;
	const int SIFT_FIXPT_SCALE = 1;
	#endif


class MySIFT
{
public:
	//////////////////////////////////////////////////////////////////////////////////////////


	/*
	 * This is the implementation of openCV for calculating SIFT Descriptor,
	 * I have just copied it out because I don't have such an interface through openCV 
	 * and I might need to apply some changes
	 * img: input image
	 * ptf: input pixel
	 * ori: angle in degrees
	 * scl: scale (but don't know yet what how to use)
	 * d: SIFT_DESCR_WIDTH (normally 4 in lowe implementaion)
	 * n: SIFT_DESCR_HIST_BINS (normally 8 in lowe implementation)
	 * dst: the descriptor array of size [n*n*d]
	 */
	static void calcSIFTDescriptor_1( const Mat& img, Point2f ptf, float ori, float scl,
								   int d, int n, float* dst )
	{
		Point pt(cvRound(ptf.x), cvRound(ptf.y));
		float cos_t = cosf(ori*(float)(CV_PI/180));
		float sin_t = sinf(ori*(float)(CV_PI/180));
		float bins_per_rad = n / 360.f;
		float exp_scale = -1.f/(d * d * 0.5f);
		float hist_width = SIFT_DESCR_SCL_FCTR * scl;
		int radius = cvRound(hist_width * 1.4142135623730951f * (d + 1) * 0.5f);
		cos_t /= hist_width;
		sin_t /= hist_width;

		int i, j, k, len = (radius*2+1)*(radius*2+1), histlen = (d+2)*(d+2)*(n+2);
		int rows = img.rows, cols = img.cols;

		AutoBuffer<float> buf(len*6 + histlen);
		float *X = buf, *Y = X + len, *Mag = Y, *Ori = Mag + len, *W = Ori + len;
		float *RBin = W + len, *CBin = RBin + len, *hist = CBin + len;

		for( i = 0; i < d+2; i++ )
		{
			for( j = 0; j < d+2; j++ )
				for( k = 0; k < n+2; k++ )
					hist[(i*(d+2) + j)*(n+2) + k] = 0.;
		}

		for( i = -radius, k = 0; i <= radius; i++ )
			for( j = -radius; j <= radius; j++ )
			{
				// Calculate sample's histogram array coords rotated relative to ori.
				// Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
				// r_rot = 1.5) have full weight placed in row 1 after interpolation.
				float c_rot = j * cos_t - i * sin_t;
				float r_rot = j * sin_t + i * cos_t;
				float rbin = r_rot + d/2 - 0.5f;
				float cbin = c_rot + d/2 - 0.5f;
				int r = pt.y + i, c = pt.x + j;

				if( rbin > -1 && rbin < d && cbin > -1 && cbin < d &&
					r > 0 && r < rows - 1 && c > 0 && c < cols - 1 )
				{
					float dx = (float)(img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1));
					float dy = (float)(img.at<sift_wt>(r-1, c) - img.at<sift_wt>(r+1, c));
					X[k] = dx; Y[k] = dy; RBin[k] = rbin; CBin[k] = cbin;
					W[k] = (c_rot * c_rot + r_rot * r_rot)*exp_scale;
					k++;
				}
			}

		len = k;
		fastAtan2(Y, X, Ori, len, true);
		magnitude(X, Y, Mag, len);
		exp(W, W, len);

		for( k = 0; k < len; k++ )
		{
			float rbin = RBin[k], cbin = CBin[k];
			float obin = (Ori[k] - ori)*bins_per_rad;
			float mag = Mag[k]*W[k];

			int r0 = cvFloor( rbin );
			int c0 = cvFloor( cbin );
			int o0 = cvFloor( obin );
			rbin -= r0;
			cbin -= c0;
			obin -= o0;

			if( o0 < 0 )
				o0 += n;
			if( o0 >= n )
				o0 -= n;

			// histogram update using tri-linear interpolation
			float v_r1 = mag*rbin, v_r0 = mag - v_r1;
			float v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
			float v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01;
			float v_rco111 = v_rc11*obin, v_rco110 = v_rc11 - v_rco111;
			float v_rco101 = v_rc10*obin, v_rco100 = v_rc10 - v_rco101;
			float v_rco011 = v_rc01*obin, v_rco010 = v_rc01 - v_rco011;
			float v_rco001 = v_rc00*obin, v_rco000 = v_rc00 - v_rco001;

			int idx = ((r0+1)*(d+2) + c0+1)*(n+2) + o0;
			hist[idx] += v_rco000;
			hist[idx+1] += v_rco001;
			hist[idx+(n+2)] += v_rco010;
			hist[idx+(n+3)] += v_rco011;
			hist[idx+(d+2)*(n+2)] += v_rco100;
			hist[idx+(d+2)*(n+2)+1] += v_rco101;
			hist[idx+(d+3)*(n+2)] += v_rco110;
			hist[idx+(d+3)*(n+2)+1] += v_rco111;
		}

		// finalize histogram, since the orientation histograms are circular
		for( i = 0; i < d; i++ )
			for( j = 0; j < d; j++ )
			{
				int idx = ((i+1)*(d+2) + (j+1))*(n+2);
				hist[idx] += hist[idx+n];
				hist[idx+1] += hist[idx+n+1];
				for( k = 0; k < n; k++ )
					dst[(i*d + j)*n + k] = hist[idx+k];
			}
		// copy histogram to the descriptor,
		// apply hysteresis thresholding
		// and scale the result, so that it can be easily converted
		// to byte array
		float nrm2 = 0;
		len = d*d*n;
		for( k = 0; k < len; k++ )
			nrm2 += dst[k]*dst[k];
		float thr = std::sqrt(nrm2)*SIFT_DESCR_MAG_THR;
		for( i = 0, nrm2 = 0; i < k; i++ )
		{
			float val = min(dst[i], thr);
			dst[i] = val;
			nrm2 += val*val;
		}
		nrm2 = SIFT_INT_DESCR_FCTR/max(std::sqrt(nrm2), FLT_EPSILON);

		for( k = 0; k < len; k++ )
		{
			dst[k] = saturate_cast<uchar>(dst[k]*nrm2);
		}
	}

	// calculate single SIFT descriptor needed for the training
	// assuming the image is already gray
	static vector<float> getSiftDescriptor(Mat &gray, KeyPoint k,int n = 4,int d = 8)
	{
		// calculate histograms
		vector<float> desc(2 * n * n * d);;

		try
		{
			calcSIFTDescriptor_1(gray,k.pt,k.angle,k.size,n,d,&(*desc.begin()));
			calcSIFTDescriptor_1(gray,k.pt,k.angle,k.size/2,n,d,&(*(desc.begin() + (n * n * d))));
		}
		catch(...)
		{
			printf("img.cols = %d x = %f\n", gray.cols, k.pt.x);
			printf("img.rows = %d y = %f\n", gray.rows, k.pt.y);
		}
		return desc;
	}

	// calculate SIF descriptors needed for the training
	static Mat getSiftDescriptors(Mat &img, vector<KeyPoint>& keypoints,int n = 4,int d = 8)
	{
		// calculate histograms
		Mat desc = Mat::zeros(keypoints.size(), 2 * n * n * d, CV_32F);;
		/// Convert it to gray
		Mat gray;
		cvtColor( img, gray, CV_RGB2GRAY );
		gray.convertTo(gray,CV_32FC1);

		for(int i = 0;i < keypoints.size();i++)
		{
			KeyPoint k = keypoints[i];
			try
			{
				calcSIFTDescriptor_1(gray,k.pt,k.angle,k.size,n,d,((float*)desc.data) + i * (2 * n * n * d) );
				calcSIFTDescriptor_1(gray,k.pt,k.angle,k.size/2,n,d,((float*)desc.data) + (i * 2 + 1) * (n * n * d));
			}
			catch(...)
			{
				printf("img.cols = %d x = %f\n", img.cols, k.pt.x);
				printf("img.rows = %d y = %f\n", img.rows, k.pt.y);
			}
		}
		return desc;
	}
};
#endif
