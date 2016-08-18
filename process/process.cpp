#include "generial.h"
using namespace cv;

void saveTrainAndTestFile(char * fileName, float**resultTrainData, float** resultTestData,int trainSize, int testSize)
{

	cout<<fileName<<endl;
	cout<<trainSize<<" is trainSize "<< testSize<<" is test size"<<endl;
	ifstream exis(fileName);//创建目标文件
	if(exis)
	{
		cout<<"file exist!"<<fileName<<endl;
		exis.close();
		return ;
	}
	ofstream out(fileName);//创建目标文件
	for (int i = 0; i < trainSize; ++i)
	{
		out<<0<<":"<<i+1<<" ";
		for (int j = 0; j < trainSize; ++j)
		{
			out<< j <<":"<< resultTrainData[i][j]<<" ";
		}
		out<<endl;
	}
	for (int i = 0; i < testSize; ++i)
	{
		out<<0<<":"<<i+1<<" ";
		for (int j = 0; j < trainSize; ++j)
		{
			out<< j <<":"<< resultTestData[i][j]<<" ";
		}
		out<<endl;
	}
	out.close();
}
void readWFromFile(char * trainVideoName,float * trainW,int index)
{
	ifstream in(trainVideoName);
	for (int i = 0; i < darwinDimension; ++i)
	{
		in>>trainW[index*darwinDimension + i];
	}
	in.close();
	return ;
}
void darWinNormalizedL2(float ** Data, int frames, int di)
{
    float sum = 0;
    for (int i = 0; i < frames; ++i)
    {
        sum = 0;
        for (int j = 0; j < di; ++j)
        {
            sum += fabs(Data[i][j]) *fabs(Data[i][j]);
        }
        sum = sqrt(sum);
        for (int j = 0; j < di; ++j)
        {
            Data[i][j] /= sum;
        }
    }
    return;
}

int main(int argc, char const *argv[])
{

	char ** trainAndTestVideoName = getFullVideoName();
	char **trainVideoName = new char*[trainNum];
	char **testVideoName = new char*[testNum];
	int trainSize = trainNum;
	int testSize = testNum;
	for (int i = 0; i < trainNum; ++i)
	{
		trainVideoName[i] = new char[filePathSize];
		strcpy(trainVideoName[i], trainAndTestVideoName[i]);
	}
	for (int i = 0; i < testNum; ++i)
	{
		testVideoName[i] = new char[filePathSize];
		strcpy(testVideoName[i], trainAndTestVideoName[i+trainNum]);
	}
	
	float * trainW = new float[trainSize*darwinDimension];
	
	
	for (int s = 0; s < trainSize; ++s)
	{
		cout<<s<<" read trainfile"<<endl;
		readWFromFile(trainVideoName[s],trainW,s);	
		
	}
	
	float * testW = new float[testSize*darwinDimension];
	for (int s = 0; s < testSize; ++s)
	{
		cout<<s<<" read testFile"<<endl;
		readWFromFile(testVideoName[s],&Label,testW,s);	
	}
	
	CvMat *trainData,*trainDataRev,*resultTrainData;
	trainData = cvCreateMat( trainSize, darwinDimension, CV_32FC1);
	trainDataRev = cvCreateMat( darwinDimension, trainSize, CV_32FC1);
	resultTrainData = cvCreateMat( trainSize, trainSize, CV_32FC1);
	cout<<"before init ======================="<<endl;
	cvInitMatHeader( trainData, trainSize, darwinDimension, CV_32FC1, trainW);
	cout<<"before transpose ======================="<<endl;
	cvTranspose(trainData,trainDataRev);
	cout<<"after transpose ======================="<<endl;
	cvMatMulAdd( trainData, trainDataRev, 0, resultTrainData);
	cout<<"after mul ======================="<<endl;
	cvReleaseMat(&trainData);

	CvMat *testData,*resultTestData;
	testData = cvCreateMat( testSize, darwinDimension, CV_32FC1);
	resultTestData = cvCreateMat( testSize, trainSize, CV_32FC1);
	cvInitMatHeader( testData, testSize, darwinDimension, CV_32FC1, testW);
	cvMatMulAdd( testData, trainDataRev, 0, resultTestData);

	cvReleaseMat(&trainDataRev);
	cvReleaseMat(&testData);

	float ** floatTrainData = new float*[trainSize];
		for (int i = 0; i < trainSize; ++i)
		{
			floatTrainData[i] = new float[trainSize];
			for (int j = 0; j < trainSize ; ++j)
			{
				floatTrainData[i][j] = CV_MAT_ELEM(* resultTrainData,float,i,j);
			}
		}
		cvReleaseMat(&resultTrainData);

	float ** floatTestData = new float*[testSize];
	for (int i = 0; i < testSize; ++i)
	{
		floatTestData[i] = new float[trainSize];
		for (int j = 0; j < trainSize ; ++j)
		{
			floatTestData[i][j] = CV_MAT_ELEM(* resultTestData,float,i,j);
		}
	}
	cvReleaseMat(&resultTestData);

	darWinNormalizedL2(floatTrainData, trainSize, trainSize);
	darWinNormalizedL2(floatTestData, testSize, trainSize);


	saveTrainAndTestFile(trainAndTestFilePath,floatTrainData,floatTestData,trainSize, testSize);
	delete []trainW;
	delete []trainLabel;
	delete []testW;
	delete []testLabel;
	
	delete c;
	return 0;
}
