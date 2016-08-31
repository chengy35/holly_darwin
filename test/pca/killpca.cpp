#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

#define DIMENTIONS	4


float Coordinates_test[DIMENTIONS]={
	0.1043,0.987,0.215,0.1156
};

#define PCA_MEAN	"mean"
#define PCA_EIGEN_VECTOR	"vector"



float* computPCAandReduce(float * data,int SAMPLE_NUM, double pcaFactor)
{
	Mat input(1,DIMENTIONS, CV_32FC1);//Test input
	for (int j=0; j<DIMENTIONS; ++j)
	{
		input.at<float>(0, j) = data[j];
	}

    PCA *pca_encoding  = new PCA();
    char *filepath = "./pca";

    FileStorage fs_r(filepath, FileStorage::READ);
    fs_r[PCA_MEAN] >> pca_encoding->mean;
    fs_r[PCA_EIGEN_VECTOR] >> pca_encoding->eigenvectors;
    fs_r.release();

    Mat output_encode(SAMPLE_NUM, pca_encoding->eigenvectors.rows, CV_32FC1);
    pca_encoding->project(input, output_encode);
    cout<<output_encode<<" =============done"<<endl;
    float * oneDimResult = new float[int(DIMENTIONS*pcaFactor*SAMPLE_NUM)+1];
    int oneIndex = 0;
    for (int i = 0; i < SAMPLE_NUM;i++) // the reduced dimension  matrix. (pca)
    {
        for (int j = 0; j < DIMENTIONS*pcaFactor;j++)
        {
            oneDimResult[oneIndex++] = output_encode.at<float>(i,j);// index is 0, for interface gmm...getAndSaveGmmModel
        }
    }
    input.release();
    output_encode.release();
    delete pca_encoding;
    //delete filepath;
    return oneDimResult;
}


int main()
{

	Mat input(1,DIMENTIONS, CV_32FC1);//Test input
	for (int j=0; j<DIMENTIONS; ++j)
	{
		input.at<float>(0, j) = Coordinates_test[j];
	}
	//Encoding

	PCA *pca_encoding = new PCA();
	FileStorage fs_r("./pca", FileStorage::READ);
	fs_r[PCA_MEAN] >> pca_encoding->mean;
	fs_r[PCA_EIGEN_VECTOR] >> pca_encoding->eigenvectors;
	fs_r.release();
	Mat output_encode(1, pca_encoding->eigenvectors.rows, CV_32FC1);
	pca_encoding->project(input, output_encode);
	cout << endl << "pca_encode:" << endl << output_encode;
	// double pcaFactor = 0.5;

	// float * reducedTRJData = computPCAandReduce(Coordinates_test,1,pcaFactor);

	// delete reducedTRJData;
	return 0;
}