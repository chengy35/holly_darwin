#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

#define DIMENTIONS	7
#define SAMPLE_NUM	31

float Coordinates[DIMENTIONS*SAMPLE_NUM]={
		 101.5,100.4,97.0,98.7,100.8,114.2,104.2
		,100.8,93.5,95.9,100.7,106.7,104.3,106.4
		,100.8,97.4,98.2,98.2,99.5,103.6,102.4
		,99.4,96.0,98.2,97.8,99.1,98.3,104.3
		,101.8,97.7,99.0,98.1,98.4,102.0,103.7
		,101.8,96.8,96.4,92.7,99.6,101.3,103.4
		,101.3,98.2,99.4,103.7,98.7,101.4,105.3
		,101.9,100.0,98.4,96.9,102.7,100.3,102.3
		,100.3,98.9,97.2,97.4,98.1,102.1,102.3
		,99.3,97.7,97.6,101.1,96.8,110.1,100.4
		,98.7,98.4,97.0,99.6,95.6,107.2,99.8
		,99.7,97.7,98.0,99.3,97.3,104.1,102.7
		,97.6,96.5,97.6,102.5,97.2,100.6,99.9
		,98.0,98.4,97.1,100.5,101.4,103.0,99.9
		,101.1,98.6,98.7,102.4,96.9,108.2,101.7
		,100.4,98.6,98.0,100.7,99.4,102.4,103.3
		,99.3,96.9,94.0,98.1,99.7,109.7,99.2
		,98.6,97.4,96.4,99.8,97.4,102.1,100.0
		,98.2,98.2,99.4,99.3,99.7,101.5,99.9
		,98.5,96.3,97.0,97.7,98.7,112.6,100.4
		,98.4,99.2,98.1,100.2,98.0,98.2,97.8
		,99.2,97.4,95.7,98.9,102.4,114.8,102.6
		,101.3,97.9,99.2,98.8,105.4,111.9,99.9
		,98.5,97.8,94.6,102.4,107.0,115.0,99.5
		,98.3,96.3,98.5,106.2,92.5,98.6,101.6
		,99.3,101.1,99.4,100.1,103.6,98.7,101.3
		,99.2,97.3,96.2,99.7,98.2,112.6,100.5
		,100.0,99.9,98.2,98.3,103.6,123.2,102.8
		,102.2,99.4,96.2,98.6,102.4,115.3,101.2
		,100.1,98.7,97.4,99.8,100.6,112.4,102.5
		,104.3,98.7,100.2,116.1,105.2,101.6,102.6
};


#define PCA_MEAN	"mean"
#define PCA_EIGEN_VECTOR	"eigen_vector"
int main()
{
//load samples
	Mat SampleSet(SAMPLE_NUM, DIMENTIONS, CV_32FC1);
	for (int i=0; i<(SAMPLE_NUM); ++i)
	{
		for (int j=0; j<DIMENTIONS; ++j)
		{
			SampleSet.at<float>(i, j) = Coordinates[i*DIMENTIONS + j];
		}
	}
//Training
	PCA *pca = new PCA(SampleSet, Mat(), CV_PCA_DATA_AS_ROW);///////////////
	//cout << "eigenvalues:" <<endl << pca->eigenvalues <<endl<<endl;
	//cout << "eigenvectors" <<endl << pca->eigenvectors << endl;


	//calculate the decreased dimensions
	int index = DIMENTIONS * 0.5;

	Mat eigenvetors_d;
	eigenvetors_d.create(index, DIMENTIONS, CV_32FC1);//eigen values of decreased dimension
	for (int i=0; i<index; ++i)
	{
		pca->eigenvectors.row(i).copyTo(eigenvetors_d.row(i));
	}
	//cout << "eigenvectors" <<endl << eigenvetors_d << endl;
	FileStorage fs_w("config.xml", FileStorage::WRITE);//write mean and eigenvalues into xml file
	fs_w << PCA_MEAN << pca->mean;
	fs_w << PCA_EIGEN_VECTOR << eigenvetors_d;
	fs_w.release();


	return 0;
}