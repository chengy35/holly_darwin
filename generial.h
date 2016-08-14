#ifndef GENERIAL
#define GENERIAL
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string.h>
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
static char* iline;
static int max_iline_len = 1024;
using namespace std;

static int st = 1;
static int datasetSize = 1708;
static int send = datasetSize;
static const int actionType = 13;
static const int filePathSize = 100;

static char *vocabDir = "../../remote/Data/Vocab/";
static char *featDir = "../../remote/Data/feats/";
static char *descriptor_path = "../../remote/Data/descriptor/";

static int cur_test_index_s = 824;
static int cur_train_index_s = 1;
static int cur_train_index_e = 823;
const int gmmSize = 256;
char* Readiline(FILE* input)
{
	int len;

	if(fgets(iline, max_iline_len, input) == NULL)
		return NULL;
	while(strrchr(iline, '\n') == NULL)
	{
		max_iline_len *= 2;
		iline = (char *) realloc(iline, max_iline_len);
		len = (int) strlen(iline);
		if(fgets(iline + len, max_iline_len - len, input) == NULL)
			break;
	}
	return iline;
}
int readLineFromFile(char * fileName)
{
	FILE *fp = fopen(fileName,"r");
	max_iline_len = 1024;
	iline = Malloc(char,max_iline_len);
	int len = 0;
	while(Readiline(fp)!=NULL)
	{
		len++;
	}
	fclose(fp);
	return len;
}

int ** readLabelFromFile()
{
	int **label = new int*[datasetSize];
	for (int i = 1; i < datasetSize; ++i)
	{
		label[i] = new int[actionType];
		for (int j = 1; j < actionType; ++j)
		{
			label[i][j] = -1;
		}
	}
	FILE *labelfile = fopen("./classLabel.txt","r");
	int iiline = 1;
	int index=1;
	int tempvalue = 0;
	while (Readiline(labelfile))
	{
		index =1;
		char * feature = strtok(iline, " \t"); // label
		tempvalue = atoi(feature);
		if (tempvalue == 1 )
		{
			label[iiline][index] = 1;
			break;
		}
		index++;
		while (1)
		{
			feature = strtok(NULL, " \t");
			tempvalue = atoi(feature);
			if (tempvalue == 1 )
			{
				label[iiline][index] = 1;
				break;
			}
			index++;
		}
		iiline++;
	}
	fclose(labelfile);
	return label;
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
#endif