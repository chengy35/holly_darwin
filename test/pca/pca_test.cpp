#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <stdlib.h>
using namespace cv;
using namespace std;

float Coordinates[32] = {
    -14.8271317103068,-3.00108550936016,1.52090778549498,3.95534842970601,
    -16.2288612441648,-2.80187433749996,-0.410815700402130,1.47546694457079,
    -15.1242838039605,-2.59871263957451,-0.359965674446737,1.34583763509479,
    -15.7031424565913,-2.53005662064257,0.255003254103276,-0.179334985754377,
    -17.7892158910100,-3.32842422986555,0.255791146332054,1.65118282449042,
    -17.8126324036279,-4.09719527953407,-0.879821957489877,-0.196675865428539,
    -14.9958877514765,-3.90753364293621,-0.418298866141441,-0.278063876667954,
    -15.5246706309866,-2.08905845264568,-1.16425848541704,-1.16976057326753};


float* computAndSavePCA(Mat pcaSet,int DIMENTIONS ,int SAMPLE_NUM, double pcaFactor,int * sizeofResult)
{
   
    PCA pca(pcaSet,Mat(),CV_PCA_DATA_AS_ROW);
   
    Mat dst = pca.project(pcaSet);//映射新空间,from this new space , find half of elements !!!
    Mat result = dst.colRange(0,DIMENTIONS*pcaFactor);//return dst(:,1:deminsion*factorsize);

    char *filepath = new char [100];
    strcpy(filepath,"./pca");


    Mat eigenvetors_d;
    int index = pcaFactor * DIMENTIONS;
    eigenvetors_d.create(index, DIMENTIONS, CV_32FC1);//eigen values of decreased dimension
    for (int i = 0; i<index; ++i)
    {
        pca.eigenvectors.row(i).copyTo(eigenvetors_d.row(i));
    }
    //cout << "eigenvectors" <<endl << eigenvetors_d << endl;
    FileStorage fs_w(filepath, FileStorage::WRITE);//write mean and eigenvalues into xml file
    fs_w << "mean" << pca.mean;
    fs_w << "vector" << eigenvetors_d;
    fs_w.release();

    cout<<"done pca====================="<<endl;
    cout<<result<<endl;
    float * oneDimResult = new float[int(DIMENTIONS*pcaFactor*SAMPLE_NUM)+1];
    int oneIndex = 0;
    for (int i = 0; i < SAMPLE_NUM;i++) // the reduced dimension  matrix. (pca)
    {
        for (int j = 0; j < DIMENTIONS*pcaFactor;j++)
        {
            oneDimResult[oneIndex++] = result.at<float>(i,j);// index is 0, for interface gmm...getAndSaveGmmModel
        }
    }
    delete[] filepath;
    *sizeofResult = oneIndex;
    return oneDimResult;
}


int main()
{
    Mat pcaSet(8, 4, CV_32FC1); //原始数据
    for (int i = 0; i < 8;i++)
    {
        for (int j = 0; j < 4;j++)
        {
            pcaSet.at<float>(i, j) = Coordinates[i*4 + j];
        }
    }
    // //cout<<pcaSet<<endl;
    // cout<<"=============="<<endl;
    // PCA pca(pcaSet,Mat(),CV_PCA_DATA_AS_ROW);
    // cout << pca.mean;//均值
    // cout << endl;

    // cout << pca.eigenvalues << endl;//特征值
    // cout << endl;

    // cout << pca.eigenvectors << endl;//特征向量

    // Mat dst = pca.project(pcaSet);//映射新空间,from this new space , find half of elements !!! 
    // cout << endl;
    // cout << dst;
    // cout << endl;

    // Mat src = pca.backProject(dst);//反映射回来
    // cout << endl;
    // cout << src;

    int b = 0;
    int *a = &b;
    double pcaFactor = 0.5;
    float * result = computAndSavePCA( pcaSet,4, 8, pcaFactor, a);
    cout<<*a<<"==============="<<endl;
    for (int i = 0; i < *a; ++i)
    {
        cout<<result[i]<<" ";
    }
    cout<<endl;
    delete result;
} 