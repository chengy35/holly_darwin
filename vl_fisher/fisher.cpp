#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <zlib.h>
#include <vector>
#include <string.h>
#include <set>
#include <iterator>
#include <algorithm>
#include <math.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
extern "C"
{
	#include <vl/generic.h>
	#include <vl/fisher.h>
}
const int TRJ_DI = 30;
const int HOG_DI = 96;
const int HOF_DI = 108;
const int MBH_DI = 192;
const int OBJ_DI = 10;
const int gmmSize = 256;
static char* iiline;
static int max_line_len = 1024;
const float pcaFactor = 0.5f;

//get config of we need. //
static int st = 1;
static int datasetSize = 1708;
static int send = datasetSize;
static const int actionType = 13;
static const int filePathSize = 100;
#define PCA_MEAN	"mean"
#define PCA_EIGEN_VECTOR	"eigen_vector"
using namespace std;
using namespace cv;
void saveDatatoFile(float* trjData,int size,char *filepath)
{
	ofstream file(filepath);
	cout<<size<<" is the size"<<endl;
	for (size_t  i = 0;  i < size; i ++) {
		file << trjData[i]<<" ";
	}
	file.close();
	cout<<"file saved !!!!!!!!!!!!!!=============="<<endl;
}
void saveVectorFile(std::vector<vector<float> > *trjData,char *filepath)
{
	ofstream file(filepath);

	for (size_t  i = 0;  i < (*trjData).size(); i ++) {
		for (int j = 0; j < TRJ_DI; ++j)
		{
			file<<(*trjData)[i][j]<<" ";
		}
		file<<endl;
	}
	file.close();
	cout<<"file saved !!!!!!!!!!!!!!=============="<<endl;
}
char * ReadLine(gzFile gzfp)
{
	int len;
	if(gzgets(gzfp, iiline, max_line_len) == NULL)
		return NULL;
	while(strrchr(iiline, '\n') == NULL)
	{
		max_line_len *= 2;
		iiline = (char*) realloc(iiline, max_line_len);
		len = (int) strlen(iiline);
		if(gzgets(gzfp, iiline + len, max_line_len - len) == NULL)
			break;
	}
	return iiline;
}
void saveFisherVectorToFile(float ** fv_trj,int num, int dimension,char *fv_file)
{
	ofstream file(fv_file);
	cout<<"save***************"<<fv_file<<endl;
	for (size_t  i = 0;  i < num; i ++) {
		for (size_t s = 0; s < dimension; s++)
		{
			file << fv_trj[i][s]<<" ";
		}
		file<<endl;
	}
	file.close();
	cout<<"file saved !!!!!!!!!!!!!!=============="<<endl;
}
void readGmmFromFile(float *data, char *fileName)
{
	cout<<"read Gmm from "<<fileName<<endl;
	ifstream _file(fileName);
	if(_file)
	{
		int index = 0;
		while(!_file.eof())
		{
			_file >> data[index++];
		}
		_file.close();
	}
	else{
		cout<<fileName<<" do not exist"<<endl;
		return ;
	}
}
void getDescriptorFromFile(char *descriptorFileName,vector<int>*obj,vector<vector<float> > *trj,vector<vector<float> > *hog,vector<vector<float> > *hof,vector<vector<float> > *mbh)
{
	printf("%s\n", "getDescriptorFromFile function to get fisher vector");
	int nLines = 0;
	max_line_len = 1024;
	iiline = Malloc(char, max_line_len);
	gzFile file = gzopen(descriptorFileName,"r");
	float value;
	int value2;
	vector<float> temp;

	while (ReadLine(file) != NULL)
	{
		//printf("%s\n","readline" );
		int j;
		char* feature;
		feature = strtok(iiline," \t");
		value2 = atoi(feature);
		obj->push_back(value2);

		for (j = 0; j < OBJ_DI-1; j++)
		{
			feature = strtok(NULL," \t"); // get rid of obj and trj
		}
		temp.clear();
		for (j = 0; j< TRJ_DI; j++)
		{
			feature = strtok(NULL," \t");
			value = atof(feature);
			temp.push_back(value);
		}
		trj->push_back(temp);
		temp.clear();
		for (j = 0; j< HOG_DI; j++)
		{
			feature = strtok(NULL," \t");
			value = atof(feature);
			value = sqrt(value);
			temp.push_back(value);
		}
		hog->push_back(temp);
		temp.clear();
		for (j = 0; j < HOF_DI; j++)
		{
			feature = strtok(NULL," \t");
			value = atof(feature);
			value = sqrt(value);
			temp.push_back(value);

		}
		hof->push_back(temp);
		temp.clear();
		for (j = 0; j < MBH_DI; j++)
		{
			feature = strtok(NULL," \t");
			value = atof(feature);
			value = sqrt(value);
			temp.push_back(value);
		}
		mbh->push_back(temp);
	}
	temp.clear();
	//printf("%s, and frame-size is ,%d \n","the end of features",(*obj).size());
	//printf("%s, %d and feature size is ,%d \n","the end of features",(*mbh).size(),(*mbh)[0].size() );
	gzclose(file);
}


float* computPCAandReduce(vector<vector<float> > *data,int startIndex, int DIMENTIONS ,int SAMPLE_NUM, double pcaFactor)
{
	//cout<<startIndex<<" is start"<<endIndex<<" is end "<< DIMENTIONS<<" is dimension"<<SAMPLE_NUM<<" is sample number"<<pcaFactor<<" is factor"<<endl;
	Mat input(SAMPLE_NUM, DIMENTIONS, CV_32FC1); //原始数据
    for (size_t i = 0; i < SAMPLE_NUM; i++)
    {
        for (int j = 0; j < DIMENTIONS;j++)
        {
            input.at<float>(i, j) = (*data)[i+startIndex][j];
        }
    }

    PCA *pca_encoding  = new PCA();
    char *filepath = new char [100];
	if(DIMENTIONS == TRJ_DI)
		strcpy(filepath,"../../remote/Data/Vocab/pcaTrjInfo.xml");
	else if(DIMENTIONS == HOG_DI)
		strcpy(filepath,"../../remote/Data/Vocab/pcaHogInfo.xml");
	else if(DIMENTIONS == HOF_DI)
		strcpy(filepath,"../../remote/Data/Vocab/pcaHofInfo.xml");
	else if(DIMENTIONS == MBH_DI)
		strcpy(filepath,"../../remote/Data/Vocab/pcaMbhInfo.xml");

	FileStorage fs_r(filepath, FileStorage::READ);
	fs_r[PCA_MEAN] >> pca_encoding->mean;
	fs_r[PCA_EIGEN_VECTOR] >> pca_encoding->eigenvectors;
	fs_r.release();

	Mat output_encode(SAMPLE_NUM, pca_encoding->eigenvectors.rows, CV_32FC1);
	pca_encoding->project(input, output_encode);

	float * oneDimResult = new float[int(DIMENTIONS*pcaFactor*SAMPLE_NUM)+1];
	int oneIndex = 0;
	for (int i = 0; i < SAMPLE_NUM;i++) // the reduced dimension  matrix. (pca)
    {
        for (int j = 0; j < DIMENTIONS*pcaFactor;j++)
        {
            oneDimResult[oneIndex++] = output_encode.at<float>(i,j);// index is 0, for interface gmm...getAndSaveGmmModel
        }
    }
	delete pca_encoding;
	return oneDimResult;
}

void saveObjDatatoFile(std::vector<int> obj, char *fileName)
{
	ofstream file(fileName);
	//cout<<obj.size()<<"===================is obj's size"<<endl;
	for (int i = 0; i < obj.size(); ++i)
	{
		file<<obj[i]<<endl;
	}
	file.close();
	return ;
}
void getRealFV(void * encTrj,float ** fv_trj, int indexLineofFisher,float * reducedTRJData,
	float * Trjmeans, int dimension, int gmmSize, float * Trjcovariances,float * Trjpriors, int numDataToEncode)
{
	    vl_fisher_encode(encTrj,VL_TYPE_FLOAT,Trjmeans,dimension, gmmSize, 
			Trjcovariances,Trjpriors,(void *)reducedTRJData, numDataToEncode,VL_FISHER_FLAG_IMPROVED);
		for (size_t i = 0; i < 2*dimension*gmmSize; i++)
		{
			fv_trj[indexLineofFisher][i] = ((float * )encTrj)[i];
		}
		vl_free(encTrj);
}
void getAndSaveFV (char *descriptorFileName, int gmmSize, float * Trjmeans, float * Trjcovariances, float * Trjpriors,
	float * Hogmeans, float * Hogcovariances,float * Hogpriors, float * Hofmeans, float * Hofcovariances,
	float * Hofpriors, float * Mbhmeans,float *  Mbhcovariances,float *  Mbhpriors,  char * feat_trj_fv_file,
	char * feat_hof_fv_file,char * feat_hog_fv_file, char * feat_mbh_fv_file)
{
	vector<int> obj;

	vector<vector<float> > trj;
	vector<vector<float> > hog;
	vector<vector<float> > hof;
	vector<vector<float> > mbh;
	obj.clear();
	trj.clear();
	hog.clear();
	hof.clear();
	mbh.clear();

	getDescriptorFromFile(descriptorFileName,&obj,&trj,&hog,&hof,&mbh);
	sort(obj.begin(), obj.end());
	
	set<int> uniqueObj (obj.begin(),obj.end());
	int frames = uniqueObj.size();

	float **fv_trj = new float* [frames];
	float **fv_hog = new float* [frames];
	float **fv_hof = new float* [frames];
	float **fv_mbh = new float* [frames];
	for (size_t i = 0; i < frames; i++) {
		fv_trj[i] = new float[(int)(2*pcaFactor*TRJ_DI*gmmSize)];
		fv_hog[i] = new float[(int)(2*pcaFactor*HOG_DI*gmmSize)];
		fv_hof[i] = new float[(int)(2*pcaFactor*HOF_DI*gmmSize)];
		fv_mbh[i] = new float[(int)(2*pcaFactor*MBH_DI*gmmSize)];
	}

	set<int>::iterator it;
	int indexLineofFisher = 0;
	
	int startindex = 0;
	int tempa,tempb,numDataToEncode,tempvalue = obj[0];
	
	
	for (it = uniqueObj.begin();it != uniqueObj.end(); it++)
	{
		tempa = startindex;
		while(tempvalue == obj[startindex])
		{
			startindex++;
		}
		tempvalue = obj[startindex];
		tempb = startindex-1;
		numDataToEncode = tempb-tempa+1;
		
		void * encTrj = vl_malloc(sizeof(float) * 2 *pcaFactor* TRJ_DI *gmmSize);
		float *reducedTRJData = computPCAandReduce(&trj,tempa,TRJ_DI,numDataToEncode,pcaFactor);
		getRealFV(encTrj,fv_trj,indexLineofFisher,reducedTRJData,Trjmeans,pcaFactor*TRJ_DI,gmmSize,Trjcovariances,Trjpriors,numDataToEncode);
		
		encTrj = vl_malloc(sizeof(float) * 2 *pcaFactor* HOG_DI *gmmSize);
		reducedTRJData = computPCAandReduce(&hog,tempa,HOG_DI,numDataToEncode,pcaFactor);
		
		getRealFV(encTrj,fv_hog,indexLineofFisher,reducedTRJData,Hogmeans,pcaFactor*HOG_DI,gmmSize,Hogcovariances,Hogpriors,numDataToEncode);
	
		encTrj = vl_malloc(sizeof(float) * 2 *pcaFactor* HOF_DI *gmmSize);
		reducedTRJData = computPCAandReduce(&hof,tempa,HOF_DI,numDataToEncode,pcaFactor);
		getRealFV(encTrj,fv_hof,indexLineofFisher,reducedTRJData,Hofmeans,pcaFactor*HOF_DI,gmmSize,Hofcovariances,Hofpriors,numDataToEncode);
		
		encTrj = vl_malloc(sizeof(float) * 2 *pcaFactor* MBH_DI *gmmSize);
		reducedTRJData = computPCAandReduce(&mbh,tempa,MBH_DI,numDataToEncode,pcaFactor);
		getRealFV(encTrj,fv_mbh,indexLineofFisher,reducedTRJData,Mbhmeans,pcaFactor*MBH_DI,gmmSize,Mbhcovariances,Mbhpriors,numDataToEncode);

		indexLineofFisher++;
	}
	cout<<" frames is "<<frames <<" ******************and indexLineofFisher "<<indexLineofFisher<<endl;
	saveFisherVectorToFile(fv_trj,frames,(int)(2*pcaFactor*TRJ_DI*gmmSize),feat_trj_fv_file);
	saveFisherVectorToFile(fv_hog,frames,(int)(2*pcaFactor*HOG_DI*gmmSize),feat_hof_fv_file);
	saveFisherVectorToFile(fv_hof,frames,(int)(2*pcaFactor*HOF_DI*gmmSize),feat_hog_fv_file);
	saveFisherVectorToFile(fv_mbh,frames,(int)(2*pcaFactor*MBH_DI*gmmSize),feat_mbh_fv_file);
}

char * getStyle(char *result,int a,int index)
{

	for (int i = 0; i < 5; ++i)
	{
		result[index++] = '0';
	}
	int j = index;
	index --;
	while(a >= 1)
	{
		int rem = a % 10;
		result[index--] = rem+'0';
		a-=rem;
		a/=10;
	}
	return result;
}
char ** getFullVideoName()
{
	char *resultStart = new char[filePathSize];
	char *resultStart2 = new char[filePathSize];
	char tempStyeStart[filePathSize] = "../../remote/Hollywood2/AVIClips/actioncliptrain";
	char tempStyeStart2[filePathSize] = "../../remote/Hollywood2/AVIClips/actioncliptest";
	strcpy(resultStart,tempStyeStart);
	strcpy(resultStart2,tempStyeStart2);

	char **fullvideoname = new char*[datasetSize];
	for (int i = 1; i < datasetSize; ++i)
	{
		fullvideoname[i] = new char[filePathSize];
		if(i < 824)
		{
			strcpy(fullvideoname[i],getStyle(resultStart,i,strlen(tempStyeStart)));
		}
		else
		{
			strcpy(fullvideoname[i],getStyle(resultStart2,i-823,strlen(tempStyeStart2)));
		}
		//cout<<fullvideoname[i]<<endl;
	}
	return fullvideoname;
}

int main(int argc, char const *argv[]) {

	char ** fullvideoname = getFullVideoName();
	const char * featDir = "../../remote/Data/feats/";
	const char *descriptor_path = "../../remote/Data/descriptor/";;
	max_line_len = 1024;
	cout<<"generate fisher vector"<<endl;
	int samples = 100000; // just for test.

	double pcaFactor = 0.5;
	char *feat_file_path  = new char[50];
	char *feat_trj_fv_file = new char[50];
	char *feat_hog_fv_file = new char[50];
	char *feat_hof_fv_file = new char[50];
	char *feat_mbh_fv_file = new char[50];

	char *descriptorFilePath = new char[50];
	char *descriptorFileName = new char[50];

	char *gmmMeansTrjFileName = new char[50];
	strcpy(gmmMeansTrjFileName, "../../remote/Data/Vocab/trj.gmmmeans");
	char *gmmCovariancesTrjFileName = new char[50];
	strcpy(gmmCovariancesTrjFileName, "../../remote/Data/Vocab/trj.gmmcovariances");
	char *gmmPriorsTrjFileName = new char[50];
	strcpy(gmmPriorsTrjFileName, "../../remote/Data/Vocab/trj.gmmpriors");
	float * Trjmeans = new float[gmmSize*TRJ_DI];
	float * Trjcovariances = new float[gmmSize*TRJ_DI];
	float * Trjpriors = new float[gmmSize];
	readGmmFromFile(Trjmeans,gmmMeansTrjFileName);
	readGmmFromFile(Trjcovariances,gmmCovariancesTrjFileName);
	readGmmFromFile(Trjpriors,gmmPriorsTrjFileName);

	char *gmmMeansHogFileName = new char[50];
	strcpy(gmmMeansHogFileName, "../../remote/Data/Vocab/hog.gmmmeans");
	char *gmmCovariancesHogFileName = new char[50];
	strcpy(gmmCovariancesHogFileName, "../../remote/Data/Vocab/hog.gmmcovariances");
	char *gmmPriorsHogFileName = new char[50];
	strcpy(gmmPriorsHogFileName, "../../remote/Data/Vocab/hog.gmmpriors");
	float * Hogmeans = new float[gmmSize*HOG_DI];
	float * Hogcovariances = new float[gmmSize*HOG_DI];
	float * Hogpriors = new float[gmmSize];
	readGmmFromFile(Hogmeans,gmmMeansHogFileName);
	readGmmFromFile(Hogcovariances,gmmCovariancesHogFileName);
	readGmmFromFile(Hogpriors,gmmPriorsHogFileName);

	char *gmmMeansHofFileName = new char[50];
	strcpy(gmmMeansHofFileName, "../../remote/Data/Vocab/hof.gmmmeans");
	char *gmmCovariancesHofFileName = new char[50];
	strcpy(gmmCovariancesHofFileName, "../../remote/Data/Vocab/hof.gmmcovariances");
	char *gmmPriorsHofFileName = new char[50];
	strcpy(gmmPriorsHofFileName, "../../remote/Data/Vocab/hof.gmmpriors");
	float * Hofmeans = new float[gmmSize*HOF_DI];
	float * Hofcovariances = new float[gmmSize*HOF_DI];
	float * Hofpriors = new float[gmmSize];
	readGmmFromFile(Hofmeans,gmmMeansHofFileName);
	readGmmFromFile(Hofcovariances,gmmCovariancesHofFileName);
	readGmmFromFile(Hofpriors,gmmPriorsHofFileName);

	char *gmmMeansMbhFileName = new char[50];
	strcpy(gmmMeansMbhFileName, "../../remote/Data/Vocab/mbh.gmmmeans");
	char *gmmCovariancesMbhFileName = new char[50];
	strcpy(gmmCovariancesMbhFileName, "../../remote/Data/Vocab/mbh.gmmcovariances");
	char *gmmPriorsMbhFileName = new char[50];
	strcpy(gmmPriorsMbhFileName, "../../remote/Data/Vocab/mbh.gmmpriors");
	float * Mbhmeans = new float[gmmSize*MBH_DI];
	float * Mbhcovariances = new float[gmmSize*MBH_DI];
	float * Mbhpriors = new float[gmmSize];
	readGmmFromFile(Mbhmeans,gmmMeansMbhFileName);
	readGmmFromFile(Mbhcovariances,gmmCovariancesMbhFileName);
	readGmmFromFile(Mbhpriors,gmmPriorsMbhFileName);

	
	send = 4;
	for (size_t i = st; i <= send; i++) {
		strcpy(feat_file_path,featDir);
		strcpy(feat_trj_fv_file,strcat(strcat(strcat(feat_file_path,"trj/"),basename(fullvideoname[i])),"-fv"));
		strcpy(feat_file_path,featDir);
		strcpy(feat_hof_fv_file,strcat(strcat(strcat(feat_file_path,"hof/"),basename(fullvideoname[i])),"-fv"));
		strcpy(feat_file_path,featDir);
		strcpy(feat_hog_fv_file,strcat(strcat(strcat(feat_file_path,"hog/"),basename(fullvideoname[i])),"-fv"));
		strcpy(feat_file_path,featDir);
		strcpy(feat_mbh_fv_file,strcat(strcat(strcat(feat_file_path,"mbh/"),basename(fullvideoname[i])),"-fv"));

	
		strcpy(descriptorFilePath,descriptor_path);
		strcpy(descriptorFileName,strcat(descriptorFilePath,basename(fullvideoname[i])));
		cout<<descriptorFileName<<endl;

		getAndSaveFV(descriptorFileName, gmmSize,Trjmeans, Trjcovariances, Trjpriors, Hogmeans, Hogcovariances,
			Hogpriors, Hofmeans, Hofcovariances, Hofpriors, Mbhmeans, Mbhcovariances, Mbhpriors,  feat_trj_fv_file,
		    feat_hof_fv_file,feat_hog_fv_file,feat_mbh_fv_file);
	}
	return 0;
}
