#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <string.h>
#include <set>
#include <iterator>
#include <algorithm>
#include <math.h>
#include "generial.h"
#define INF HUGE_VAL
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

using namespace std;
const int DIMENSION = 30*gmmSize;

void readDarWinfromFile(char * fvFilePath, float **data, int index)
{
    ifstream file(fvFilePath);
    for (int i = 0; i < 2*DIMENSION*gmmSize; ++i)
    {
       file >> data[index][i];
       data[index][i] =sqrt(data[index][i]);
    }
    file.close();
    return ;
}
int main(int argc, char const *argv[])
{
    char **fullvideoname = getFullVideoName();
    int st = 1;
    int send = 4;
    int gmmSize = 256;
    int trainSize  = 823;
    int testSize = 884;
    char *feature_out = "../../remote/Data/feats/trj/";
    char *wFilePath = new char[50]; 

    int all_video = 4;//you should replace it with dataSetSize;
    float ** all_data_cell = new float*[send-st+1];
    for (int i = 0; i < all_video ; ++i)
    {
        all_data_cell[i] = new float[2*DIMENSION*gmmSize];
    }
    for (int i = st; i <= send ; ++i)
    {
        strcpy(wFilePath,"../../remote/Data/feats/w/");
        strcat(wFilePath,basename(fullvideoname[i]));
        strcat(wFilePath,"-w");
        readDarWinfromFile(wFilePath,all_data_cell,i-1);
    }

    iline = Malloc(char,max_iline_len);
    int ** classid = readLabelFromFile();
    int *trn_indx = new int[trainSize];
    int *test_indx = new int[testSize];
    for (int i = 0; i < trainSize ; ++i)
    {
        trn_indx[i] = i+1;
    }
    for (int i = 0; i < testSize ; ++i)
    {
        test_indx[i] = i+824;
    }

    return 0;
}  