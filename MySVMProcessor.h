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


#ifndef MySVMProcessor_H
#define MySVMProcessor_H

#include "MyTesting.h"

// class to provide processing for SVM
class MySVMProcessor
{
	svm_model* model;
	// range parameters
	double miny,maxy;
	vector<double> minv;
	vector<double> maxv;
public:
	// with fid number
	MySVMProcessor(string name,int fidNum)
	{
		model = Loading::loadSVMModelNumber(fidNum,name);
		Loading::loadRange(miny,maxy,minv,maxv,fidNum,name);
	}
	// general SVM model
	MySVMProcessor(string name)
	{
		model = Loading::loadSVMModel(name);
		Loading::loadGeneralRange(miny,maxy,minv,maxv,name);
	}
	double calcResponse(vector<float>& inp)
	{
		return Testing::calcResponseOnePoint(model, inp,inp.size(),miny,maxy,minv,maxv);
	}
	double calcResponse(vector<double>& inp)
	{
		return Testing::calcResponseOnePoint(model, inp,inp.size(),miny,maxy,minv,maxv);
	}
	~MySVMProcessor()
	{
		svm_free_and_destroy_model(&model);
	}

};

class MySVMS
{
	vector<MySVMProcessor*> models;
public:
	MySVMS(string name)
	{
		// loading ranges and SVM models
		for(int i = 0;i < 29;i++)
		{
			models.push_back(new MySVMProcessor(name,i));
		}
	}
	double calcResponse(vector<float>& inp,int fidInd)
	{
		if(fidInd >= 0 && fidInd < models.size())
		{
			return models[fidInd]->calcResponse(inp);
		}
		return 0;
	}
	~MySVMS()
	{
		for(int i = 0;i < models.size();i++)
		{
			delete models[i];
		}
	}
};

#endif
